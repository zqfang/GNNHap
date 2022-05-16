import os, sys, json,uuid
import pandas as pd

if 'WKDIR' in config:
    WKDIR = config['WKDIR'] # input by --config TRAIT_IDS="ids.txt"
else:
    WKDIR = config['HBCGM']['WORKSPACE']

workdir: WKDIR

HBCGM_BIN = config['HBCGM']['BIN']
STRAINS = sorted(config['STRAINS'])
# INPUT_FILES
# MESH_DICT = {
#  '10806':'D003337',
#  '10807':'D020712',
#  '10813':'D020712',
#  '26721': 'D017948',
#  '50243':'D002331',
#  '9904': 'D008076',
#  '9904': "D064420",
# #  'irinotecan': 'D000077146',
# #  'Cocaine':'D007858',
# #  'D018919':'Neovascularization, Physiologic',
# #  'D009389' :'Neovascularization, Pathologic',
# #   'D043924' : 'Angiogenesis Modulating Agents',
# #   'D003315': 'Cornea',
# }

if 'MESH_TERMS' in config:
    MESH_TERMS = config['MESH_TERMS'].strip(",") ## input by --config MESH_TERMS="D003337"
else:
    MESH_TERMS = ""

if 'UUID' in config:
    UUID = config['UUID'] 

if 'STRAIN' in config:
    inStrain=config['STRAIN'].strip(",").split(",")
if 'PHENO' in config:
    inPheno = config['PHENO'].strip(",").split(",")

MESH_DICT = {UUID: MESH_TERMS}
# MPD2MeSH = 'D018919,D009389,D043924,D003315'
# MPD2MeSH = "/data/bases/shared/haplomap/PELTZ_20210609/mpd2mesh.json"

# GNNHAP and bundle
GNNHAP =  config['HBCGM']['GNNHAP']
GNNHAP_BUNDLE =config['HBCGM']['GNNHAP_BUNDLE']

GENETIC_REL = config['HBCGM']['GENETIC_REL']
MPD2MeSH = config['HBCGM']['MPD2MeSH'] # json file
# eblock input
STRAIN_ANNO = config['HBCGM']['STRAIN_ANNO']
SNPDB = config['HBCGM']['VAR_DIR']
VCF_DIR = config['HBCGM']['VCF_DIR']
VEP_DIR = config['HBCGM']['VEP_DIR']
# gene expression file
GENE_EXPRS = config['HBCGM']['GENE_EXPRS']
CHROMOSOMES = [str(i) for i in range (1, 20)] + ['X'] # NO 'Y'

### OUTPUT HBCGM
FINAL_OUT = f"MPD_{UUID}_snp.results.txt"
FINAL_OUT_INDEL = f"MPD_{UUID}_indel.results.txt"

### OUTPUT GNNHAP
if MESH_TERMS != "":
    FINAL_OUT = f"MPD_{UUID}_snp.results.mesh.txt"
    FINAL_OUT_INDEL = f"MPD_{UUID}_indel.results.mesh.txt"

rule target:
    input: FINAL_OUT, FINAL_OUT_INDEL


rule trait_data:
    output: f"trait.{UUID}.txt"
    run:
        with open(output[0],'w') as out:
            for s, v in zip(inStrain, inPheno):
                if s == "": continue
                out.write(f"{s}\t{v}\n")

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
        os.path.join(SNPDB, "chr{i}.snp.txt"),
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
        "-q {params.qual} -d {params.het} -m {params.mq} -b {params.sb} "
        "-s {input.strain} -v > {log}"



rule Indel2NIEHS:
    input:  
        strain = "strain.order.snpdb.txt",
        vcf = os.path.join(VCF_DIR, "chr{i}.vcf.gz"),
    output: 
        os.path.join(SNPDB,"chr{i}.indel.txt")
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
        "-q {params.qual} -d {params.het} -m {params.mq} -b {params.sb} -t INDEL "
        "-s {input.strain} -v > {log}"

rule unGZip:
    input: os.path.join(VEP_DIR, "chr{i}.pass.vep.txt.gz"),
    output: temp(os.path.join(VEP_DIR, "chr{i}.pass.vep.txt"))
    shell:
        "zcat {input} > {output}"

rule annotateSNPs:
    input: 
        vep = os.path.join(VEP_DIR, "chr{i}.pass.vep.txt"),
        strains = "trait.{ids}.txt",
    output: "MPD_{ids}/Annotation/chr{i}.snp.genename.txt"
    params:
        bin = HBCGM_BIN,
    shell:
        "{params.bin}/haplomap annotate -t snp -s {input.strains} -o {output} {input.vep} "

rule annotateINDELs:
    input: 
        vep = os.path.join(VEP_DIR, "chr{i}.pass.vep.txt"),
        strains = "trait.{ids}.txt",
    output: "MPD_{ids}/Annotation/chr{i}.indel.genename.txt"
    params:
        bin = HBCGM_BIN,
    shell:
        "{params.bin}/haplomap annotate -t indel -s {input.strains} -o {output} {input.vep} "

# find haplotypes
rule eblocks:
    input: 
        snps = os.path.join(SNPDB,"chr{i}.snp.txt"),
        indels = os.path.join(SNPDB,"chr{i}.indel.txt"),
        gene_anno = "MPD_{ids}/Annotation/chr{i}.snp.genename.txt",
        gene_annoi = "MPD_{ids}/Annotation/chr{i}.indel.genename.txt",
        strains = "trait.{ids}.txt",
    output: 
        hb = "MPD_{ids}/chr{i}.snp.hblocks.txt",
        hbi = "MPD_{ids}/chr{i}.indel.hblocks.txt",
        xhb = temp("MPD_{ids}/chr{i}.snp.hblocks.temp.txt"),
        xhbi = temp("MPD_{ids}/chr{i}.indel.hblocks.temp.txt"),
    params:
        bin = HBCGM_BIN,
    log: "logs/MPD_{ids}.chr{i}.eblocks.log"
    run:
        shell("{params.bin}/haplomap eblocks -a {input.snps} -g {input.gene_anno} "
        "-s {input.strains} -p {output.xhb} "
        "-o {output.hb} -v > {log}")
        shell("{params.bin}/haplomap eblocks -a {input.indels} -g {input.gene_annoi} "
        "-s {input.strains} -p {output.xhbi} "
        "-o {output.hbi} -v > {log}")

# statistical testing with trait data       
rule ghmap:
    input: 
        hb = "MPD_{ids}/chr{i}.snp.hblocks.txt",
        hbi = "MPD_{ids}/chr{i}.indel.hblocks.txt",
        trait = "trait.{ids}.txt",
        gene_exprs = GENE_EXPRS,
        rel = GENETIC_REL,
    output: 
        snps="MPD_{ids}/chr{i}.snp.results.txt",
        indels="MPD_{ids}/chr{i}.indel.results.txt",
    params:
        bin = HBCGM_BIN,
        cat = "trait.{ids}.categorical"
    log: "logs/MPD_{ids}.chr{i}.ghmap.log"
    run:
        categorical = "-c" if os.path.exists(params.cat) else ''
        cats = "_catogorical" if os.path.exists(params.cat) else ''
        cmd = "{params.bin}/haplomap ghmap %s "%categorical +\
              "-e {input.gene_exprs} -r {input.rel} " +\
              "-p {input.trait} -b {input.hb} -o {output.snps} " +\
              "-n MPD_{wildcards.ids}%s -a -v > {log}"%cats
        shell(cmd)

        cmd = "{params.bin}/haplomap ghmap %s "%categorical +\
                "-e {input.gene_exprs} -r {input.rel} " +\
                "-p {input.trait} -b {input.hbi} -o {output.indels} " +\
                "-n MPD_{wildcards.ids}_indel%s -a -v > {log}"%cats
        shell(cmd)


rule ghmap_aggregate:
    input: 
        res = ["MPD_{ids}/chr%s.snp.results.txt"%c for c in CHROMOSOMES] 
    output: temp("MPD_{ids}_snp.results.txt")
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


rule ghmap_aggregate_indel:
    input: 
        res = ["MPD_{ids}/chr%s.indel.results.txt"%c for c in CHROMOSOMES],
    output: temp("MPD_{ids}_indel.results.txt")
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
        res = "MPD_{ids}_snp.results.txt",
    output: "MPD_{ids}_snp.results.mesh.txt"
    params: 
        # mesh = lambda wildcards: MESH_DICT[wildcards.ids]
        res_dir = WKDIR,
        gnnhap = GNNHAP,
        bundle = GNNHAP_BUNDLE,
        json = lambda wildcards: MESH_DICT[wildcards.ids],
    threads: 24
    shell:
        "/home/fangzq/miniconda/envs/fastai/bin/python {params.gnnhap}/GNNHap/predict.py "
        "--bundle {params.bundle} " 
        "--hbcgm_result_dir {params.res_dir} "
        "--mesh_terms {params.json} --num_cpus {threads} "

rule mesh_indel:
    input: 
        res = "MPD_{ids}_indel.results.txt",
    output: "MPD_{ids}_indel.results.mesh.txt"
    params: 
        # mesh = lambda wildcards: MESH_DICT[wildcards.ids]
        res_dir = WKDIR,
        gnnhap = GNNHAP,
        bundle = GNNHAP_BUNDLE,
        json = lambda wildcards: MESH_DICT[wildcards.ids],
    threads: 24
    shell:
        "/home/fangzq/miniconda/envs/fastai/bin/python {params.gnnhap}/GNNHap/predict.py "
        "--bundle {params.bundle} " 
        "--hbcgm_result_dir {params.res_dir} "
        "--mesh_terms {params.json} --num_cpus {threads} "