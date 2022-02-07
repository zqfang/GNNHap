import os, sys, json
import pandas as pd

#WKDIR = config['HBCGM']['WORKSPACE']

WKDIR = "/home/fangzq/github/GNNHap/example"
workdir: WKDIR

HBCGM_BIN = config['HBCGM']['BIN']
STRAINS = sorted(config['STRAINS'])
# INPUT_FILES
MESH_DICT = {
#  '10806':'D003337',
#  '10807':'D020712',
#  '10813':'D020712',
#  '26721': 'D017948',
#  '50243':'D002331',
#  '9904': 'D008076',
#  'Haloperidol': "D064420",
#  'irinotecan': 'D000077146',
#  'Cocaine':'D007858',
 'D018919':'Neovascularization, Physiologic',
 'D009389' :'Neovascularization, Pathologic',
  'D043924' : 'Angiogenesis Modulating Agents',
  'D003315': 'Cornea',
}
MPD2MeSH = 'D018919,D009389,D043924,D003315'
# MPD2MeSH = "/data/bases/shared/haplomap/PELTZ_20210609/mpd2mesh.json"

# binaries
# GNNHAP and bundle
GNNHAP =  "/home/fangzq/github/GNNHap"
GNNHAP_BUNDLE = "/data/bases/fangzq/Pubmed/bundle" #"/home/fangzq/github/GNNHap/bundle"

# path to SNP database
VCF_DIR = "/data/bases/shared/haplomap/PELTZ_20210609/VCFs"
# PATH to VEP annotations for all genes
VEP_DIR = "/data/bases/shared/haplomap/PELTZ_20210609/VEP"

# genetic relation file from PLink output
GENETIC_REL =  "/data/bases/shared/haplomap/PELTZ_20210609/mouse54_grm.rel"
# gene expression file
GENE_EXPRS = "/data/bases/shared/haplomap/PELTZ_20210609/mus.compact.exprs.txt"

### OUTPUT

CHROMOSOMES = [str(i) for i in range (1, 20)] + ['X'] # NO 'Y'
# output files
# SNPDB = expand("SNPs/chr{i}.txt", i=CHROMOSOMES)

MESH = "MPD_000.results.mesh.txt"

rule target:
    input: MESH


rule strainOrder:
    output: "strain.order.snpdb.txt"
    run:
        with open(output[0], 'w') as s:
            s.write("\n".join(STRAINS) +"\n")
        
rule snp2NIEHS:
    input:  
        strain = "strain.order.snpdb.txt",
        vcf = os.path.join(VCF_DIR, "chr{i}.vcf.gz"),
    output: 
        "SNPs/chr{i}.txt",
    params:
        qual = config['BCFTOOLS']['qual'], 
        het = config['BCFTOOLS']['phred_likelihood_diff'],
        ad = config['BCFTOOLS']['allele_depth'],
        ratio = config['BCFTOOLS']['allele_mindepth_ratio'],
        mq = config['BCFTOOLS']['mapping_quality'],
        sb = config['BCFTOOLS']['strand_bias_pvalue'], 
        BIN = HBCGM_BIN# path to haplomap binary
    log: "logs/chr{i}.snp2niehs.log"
    shell:
        "bcftools view -f .,PASS -v snps {input.vcf} | "
        "{params.BIN}/haplomap convert -o {output} -a {params.ad} -r {params.ratio} "
        "-q {params.qual} -p {params.het} -m {params.mq} -b {params.sb} "
        "-s {input.strain} -v > {log}"



rule Indel2NIEHS:
    input:  
        strain = "strain.order.snpdb.txt",
        vcf = os.path.join(VCF_DIR, "VCFs/{chr}.indel.vcf.gz"),
    output: 
        "INDELs/chr{i}.indel.txt"
    params:
        qual = config['BCFTOOLS']['qual'], 
        het = config['BCFTOOLS']['phred_likelihood_diff'],
        ad = config['BCFTOOLS']['allele_depth'],
        ratio = config['BCFTOOLS']['allele_mindepth_ratio'],
        mq = config['BCFTOOLS']['mapping_quality'],
        sb = config['BCFTOOLS']['strand_bias_pvalue'], 
        BIN = HBCGM_BIN# path to haplomap binary
    log: "logs/chr{i}.snp2niehs.log"
    shell:
        "bcftools view -f .,PASS -v indels {input.vcf} | "
        "{params.BIN}/haplomap convert -o {output} -a {params.ad} -r {params.ratio} "
        "-q {params.qual} -p {params.het} -m {params.mq} -b {params.sb} -t INDEL "
        "-s {input.strain} -v > {log}"

rule unGZip:
    input: os.path.join(VEP_DIR, "chr{i}.pass.vep.txt.gz"),
    output: temp(os.path.join(VEP_DIR, "chr{i}.pass.vep.txt"))
    shell:
        "zcat {input} > {output}"

rule annotateSNPs:
    input: 
        vep = os.path.join(VEP_DIR, "chr{i}.pass.vep.txt"),
        strains = "MPD_{ids}/trait.{ids}.txt",
    output: "MPD_{ids}/chr{i}.genename.txt"
    params:
        bin = HBCGM_BIN,
    shell:
        "{params.bin}/haplomap annotate -t snp -s {input.strains} -o {output} {input.vep} "

rule annotateINDELs:
    input: 
        vep = os.path.join(VEP_DIR, "chr{i}.pass.vep.txt"),
        strains = "MPD_{ids}/trait.{ids}.txt",
    output: "MPD_{ids}/chr{i}.indel.genename.txt"
    params:
        bin = HBCGM_BIN,
    shell:
        "{params.bin}/haplomap annotate -t indel -s {input.strains} -o {output} {input.vep} "

# find haplotypes
rule eblocks:
    input: 
        snps = os.path.join(SNPDB_DIR, "chr{i}.txt"),
        gene_anno = "MPD_{ids}/chr{i}.genename.txt",
        strains = "MPD_{ids}/trait.{ids}.txt",
    output: 
        hb = "MPD_{ids}/chr{i}.hblocks.txt",
        snphb ="MPD_{ids}/chr{i}.snp.hblocks.txt"
    params:
        bin = HBCGM_BIN,
    log: "logs/MPD_{ids}.chr{i}.eblocks.log"
    shell:
        "{params.bin}/haplomap eblocks -a {input.snps} -g {input.gene_anno} "
        "-s {input.strains} -p {output.snphb} "
        "-o {output.hb} -v > {log}"

# statistical testing with trait data       
rule ghmap:
    input: 
        hb = "MPD_{ids}/chr{i}.hblocks.txt",
        trait = "MPD_{ids}/trait.{ids}.txt",
        gene_exprs = GENE_EXPRS,
        rel = GENETIC_REL,
    output: "MPD_{ids}/chr{i}.results.txt"
    params:
        bin = HBCGM_BIN,
        cat = "MPD_{ids}/trait.{ids}.categorical"
    log: "logs/MPD_{ids}.chr{i}.ghmap.log"
    run:
        categorical = "-c" if os.path.exists(params.cat) else ''
        cats = "_catogorical" if os.path.exists(params.cat) else ''
        cmd = "{params.bin}/haplomap ghmap %s "%categorical +\
              "-e {input.gene_exprs} -r {input.rel} " +\
              "-p {input.trait} -b {input.hb} -o {output} " +\
              "-n MPD_{wildcards.ids}%s -a -v > {log}"%cats
        shell(cmd)


rule ghmap_aggregate:
    input: 
        res = ["MPD_{ids}/chr%s.results.txt"%c for c in CHROMOSOMES]
    output: temp("MPD_{ids}.results.txt")
    run:
        # read input
        dfs = []
        for p in input.res:
            case = pd.read_table(p, skiprows=5, dtype=str)
            dfs.append(case)
        result = pd.concat(dfs)
        # read header, first 5 row
        headers = []
        with open(p, 'r') as r:
            for i, line in enumerate(r):
                headers.append(line)
                if i >= 4: break 

        if os.path.exists(output[0]): os.remove(output[0])
        # write output
        with open(output[0], 'a') as out:
            for line in headers:
                out.write(line)
            ## Table
            result.to_csv(out, sep="\t", index=False)


rule mesh:
    input: 
        res = "MPD_{ids}.results.txt",
    output: "MPD_{ids}.results.mesh.txt"
    params: 
        # mesh = lambda wildcards: MESH_DICT[wildcards.ids]
        res_dir = WKDIR,
        gnnhap = GNNHAP,
        bundle = GNNHAP_BUNDLE,
        json = MPD2MeSH,
    threads: 24
    shell:
        "/home/fangzq/miniconda/envs/fastai/bin/python {params.gnnhap}/GNNHap/predict.py "
        "--bundle {params.bundle} " 
        "--hbcgm_result_dir {params.res_dir} "
        "--mesh_terms {params.json} --num_cpus {threads} "