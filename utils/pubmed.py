import gzip, json, os, sys

import xml.etree.cElementTree as ET
from io import StringIO
from Bio import Entrez
from bioservices.uniprot import UniProt 
import requests
import numpy as np
import pandas as pd
import networkx as nx
import spacy

class PubMed:
    def __init__(self, db='pubmed', pmid="", retmode="xml", ncbi_api_key=None):
        """
        Biopython.Entrez wrapper for getting metadata of pubmed articals
        """
        self.db = db
        self.pmids = pmid
        self.retmode = retmode
        self.api_key = ncbi_api_key
        
        if Entrez.email is None:
            # need to set who am i
            Entrez.email = 'fztsing@126.com'          
        if ncbi_api_key is not None:
            # my personal api key: "5a71da316f173953b53a497fbb39c0a32308"
            # default 3 queries per second, with api_key 10 per second
            Entrez.api_key = ncbi_api_key
        
        self.results = {}
        
    def efectch(self, pmid=""):
        self.pmid = pmid
        # efectch wrapper 
        handle = Entrez.efetch(db=self.db, id=pmid, retmode=self.retmode)
        xml_data = Entrez.read(handle) # return a dict object
        handle.close()  
        self._pmid_data = xml_data
        # number of input articles
        self._pmid_cnt = len(pmid.split(","))
        #return xml_data
    def parse_meshterm(self):
        
        articles = self._pmid_data['PubmedArticle']
        # MedlineCitation ['OtherAbstract', 'SpaceFlightMission', 
        # 'KeywordList', 'OtherID', 'CitationSubset', 'GeneralNote', 
        # 'PMID', 'DateCompleted', 'DateRevised', 'Article', 
        # 'MedlineJournalInfo', 'ChemicalList', 'MeshHeadingList']
        for i, article in enumerate(articles):         
            # Article -> dict_keys(['ArticleDate', 'ELocationID', 'Language', 
            #'Journal', 'ArticleTitle', 'Pagination', 'Abstract', 
            #'AuthorList', 'GrantList', 'PublicationTypeList'])
            metainfo = {}
            pmid = str(article['MedlineCitation']['PMID'])
            abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText']
            title = article['MedlineCitation']['Article']['ArticleTitle']
            
            #metainfo = article['MedlineCitation']['Article']
            terms = article['MedlineCitation']['MeshHeadingList']
            #metainfo['MeshHeadingList'] = terms
            metainfo['Title'] = title
            metainfo['Abstract'] = abstract
            metainfo['MeshHeadingList'] = self._parse_meshterm(terms)
            self.results[pmid]= metainfo   
    
    def _parse_meshterm(self, mhlist: list ):
        meshterms = []
        for d in mhlist:
            qterms = d['QualifierName']
            dterms = str(d['DescriptorName'])
            if qterms: 
                dterms = "/".join( [dterms] + [str(t) for t in qterms])
            meshterms.append(dterms)
        return meshterms   


    def article_links(self, start_date, end_date = '3000'):
        """
        start_date, end_date = 'YYYY/MM/DD'
        returns a list of PubMedCentral links and a 2nd list of DOI links
        """
        #get all articles in certain date range, in this case 5 articles which will be published in the future
        handle = Entrez.esearch(db="pubmed", term='("%s"[Date - Publication] : "%s"[Date - Publication]) ' %(start_date, end_date))
        records = Entrez.read(handle)

        #get a list of Pubmed IDs for all articles
        idlist = ','.join(records['IdList'])
        handle = Entrez.efetch("pubmed", id=idlist, retmode="xml")
        records = Entrez.parse(handle)

        pmc_articles = []
        doi = []
        for record in records:
            #get all PMC articles
            if record.get('MedlineCitation'):
                if record['MedlineCitation'].get('OtherID'):
                    for other_id in record['MedlineCitation']['OtherID']:
                        if other_id.title().startswith('Pmc'):
                            pmc_articles.append('http://www.ncbi.nlm.nih.gov/pmc/articles/%s/pdf/' % (other_id.title().upper()))
            #get all DOIs
            if record.get('PubmedData'):
                if record['PubmedData'].get('ArticleIdList'):
                    for other_id in record['PubmedData']['ArticleIdList']:
                        if 'doi' in other_id.attributes.values():
                            doi.append('http://dx.doi.org/' + other_id.title())


        return pmc_articles, doi

class XMLParser:
    def __init__(self, xml=None):
        if xml is None: return

        if xml.endswith(".gz"):
            xml_handle = gzip.open(xml,'r')
        else:
            xml_handle = open(xml,'r')

        xtree = ET.parse(xml_handle)
        self.xroot = xtree.getroot()
        xml_handle.close() 

    def parse(self):
        pass

class PubMedXMLPaser(XMLParser):
    def __init__(self, xml):
        """
        download pubmed xml data from 
            ftp.ncbi.nlm.nih.gov/pubmed/baseline/*xml.gz
            ftp.ncbi.nlm.nih.gov/pubmed/updatefiles/*xml.gz
        """
        super(PubMedXMLPaser, self).__init__(xml)

    def parse(self): 
        """Parse the input XML file and store the result in a pandas 
        DataFrame with the given columns. 
        """
        metadata = {}
        for particle in self.xroot.iter('PubmedArticle'): ## iteral all
            art = {}
            # get PUBMED ID
            # Element.findall() finds only elements with a tag which are direct children of the current element
            # Element.find() finds the first child with a particular tag, and Element.text accesses the element’s text content. 
            # Element.get() accesses the element’s attributes:
            #root.findall("./country/neighbor")
            pmid = particle.find('./MedlineCitation/PMID').text
            pmid2 = particle.find("./PubmedData/ArticleIdList/ArticleId[@IdType='pubmed']").text
            art['PMID'] = pmid2
            doi = particle.find("./PubmedData/ArticleIdList/ArticleId[@IdType='doi']")
            lang = particle.find("./MedlineCitation/Article/Language").text ## do we need to worry about this if only the abstract is needed
            art['Language'] = lang
            
            year = particle.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
            if year is not None:
                art['Year'] = year.text
            
            month = particle.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Month")
            if month is not None:
                art['Month'] = month.text
            
            if doi is not None:
                art['DOI'] = doi.text
            title = particle.find('./MedlineCitation/Article/ArticleTitle')
            if title is not None:
                art['Title'] = title.text
            abstract = particle.find("./MedlineCitation/Article/Abstract/AbstractText")
            if abstract is not None:
                art['Abstract'] = abstract.text
            # we might want to only select reseach articles. so keep a record for publication type
            pub_type = particle.findall("./MedlineCitation/Article/PublicationTypeList/PublicationType")
            pub = {p.get("UI"): p.text for p in pub_type }
            art['PublicationType'] = pub
            # get mesh
            mesh_list = particle.findall('./MedlineCitation/MeshHeadingList/MeshHeading')
            if mesh_list: # if not empty
                mlist = [] 
                for mesh in mesh_list:
                    D = {}
                    descriptor = mesh.find('DescriptorName')
                    if descriptor is not None:
                        D['DescriptorName'] = descriptor.text
                        D['DescriptorUI'] = descriptor.get('UI')
                        D['DescriptorMajorTopicYN'] = descriptor.get("MajorTopicYN")
                    qualifier = mesh.find("QualifierName")
                    if qualifier is not None:
                        D['QualifierName'] = qualifier.text
                        D['QualifierUI'] = descriptor.get('UI')
                        D['QualifierMajorTopicYN'] = descriptor.get("MajorTopicYN")
                    mlist.append(D)
                art['MeSH'] = mlist
            # get chemicalList
            chem_list = particle.findall('./MedlineCitation/ChemicalList/Chemical')
            if chem_list: # if not empty    
                clist = []
                for chem in chem_list:
                    C = {}
                    C['RegistryNumber'] = chem.find('RegistryNumber').text
                    substance = chem.find('NameOfSubstance')
                    C['SubstanceName'] = substance.text
                    C['SubstanceUI'] = substance.get("UI")
                    clist.append(C)
                art['Chemical'] = clist

            metadata[pmid] = art
            
        return metadata

class MeSHXMLParser(XMLParser):
    def __init__(self, xml):
        """
        Download bulk data (xml format) from here: https://www.nlm.nih.gov/databases/download/mesh.html

        - extract descriptor and its uid (this is what we need)
        - extract qualifier and its uid

        MeSH tree: hierachical tree. Treeview here: https://meshb.nlm.nih.gov/treeView       
        """
        super(MeSHXMLParser, self).__init__(xml) 
        
    def parse(self):
        """
        return Dict with keys:
            DescriptorClass: 1,2,3,4. May only need class 1? Explained here -> https://www.nlm.nih.gov/mesh/intro_record_types.html
            Treenums: location in MeSH hierachical tree. Treeview here: https://meshb.nlm.nih.gov/treeView
            Concepts: ?
            
        """
        descriptors = {}
        for rec in self.xroot.iter("DescriptorRecord"):
            desp = {}
            desp['DescriptorClass'] = rec.get("DescriptorClass")
            UI = rec.find("DescriptorUI").text
            desp['DescriptorUI'] = UI
            desp['DescriptorName'] = rec.find("DescriptorName/String").text
   
            treenums = rec.findall("TreeNumberList/TreeNumber")
            trnums = []
            for trn in treenums: 
                trnums.append(trn.text)
            desp['TreeNums'] = trnums

            concept = []
            concepts = rec.findall("ConceptList/Concept")
            for cept in concepts:
                c = {}
                c['UI'] = cept.find("ConceptUI").text
                c['Name'] = cept.find("ConceptName/String").text
                if cept.find("CASN1Name") is not None:
                    c['CASN1Name'] = cept.find("CASN1Name").text
                if cept.find("RegistryNumber") is not None:
                    c['RegistryNumber'] = cept.find("RegistryNumber").text
                if cept.find("ScopeNote") is not None:
                    c['ScopeNote'] = cept.find("ScopeNote").text
                if cept.find("TermList") is not None:
                    termlist = cept.findall("TermList/Term")
                    termall = []
                    for term in termlist:
                        term_entry = {}
                        term_entry['TermUI'] = term.find("TermUI").text
                        term_entry['Name'] = term.find("String").text
                        termall.append(term_entry)
                    c['Term'] = termall
                concept.append(c)
                
            desp['Concept'] = concept

            descriptors[UI] = desp
        return descriptors


class UniProtXMLParser(XMLParser):
    def __init__(self, xml=None):
        """
        Parse Uniprot database online or
        download the database from here: 
        
        ftp://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.xml.gz
        """
        super(UniProtXMLParser, self).__init__(xml)
        self.ns = {"xmlns": "http://uniprot.org/uniprot",
           "xsi":"http://www.w3.org/2001/XMLSchema-instance",
           "schemaLocation":"http://uniprot.org/uniprot http://www.uniprot.org/support/docs/uniprot.xsd"}
        self.uniprot = UniProt()
        # if xml is None: return      
        # if xml.endswith(".gz"):
        #     xml_handle = gzip.open(xml,'r')
        # else:
        #     xml_handle = open(xml,'r')

        # xtree = ET.parse(xml_handle)
        # self.xroot = xtree.getroot()
        # xml_handle.close() 

        
    def mapping(self, fr="P_ENTREZGENEID", to='ID',query=['11302','11304']):
        """
        see self.uniprot._mapping
        """
        ## ID mapping, see ._mapping 
        self.maps = self.uniprot.mapping(fr=fr, to=to, query=query)
        # get the first entry is enought
        return self.maps
    
    def search(self, accession):
        """
        Uniprot accession number
        """
        esayXML = self.uniprot.retrieve(accession, frmt='xml') # 1000X faster
        # self.xroot = ET.fromstring(self.uniprot.search(accession, frmt='xml'))
        metadata = self.parse(esayXML.root)
        #metadata['query_accession'] = accession
        return metadata
    
    def searchall(self, accessions: list):
        metadata = {}
        for acc in accessions:
            meta = self.search(acc)
            metadata.update(meta)
        return metadata
            
    def parse(self, xroot):
        """
        If the XML input has namespaces, tags and attributes with prefixes 
        in the form prefix:sometag get expanded to {uri}sometag where the prefix 
        is replaced by the full URI. Also, if there is a default namespace, 
        that full URI gets prepended to all of the non-prefixed tags.
        
        """
        metadata = {}
        entries = xroot.findall("xmlns:entry",self.ns)
        for entry in entries:
            meta = {}
            name = entry.find("xmlns:name", namespaces=self.ns).text
            meta['name'] = name
            meta['accession'] = [a.text for a in entry.findall("xmlns:accession",self.ns) ]
            meta['gene_names'] =[g.text for g in  entry.findall("xmlns:gene/xmlns:name", self.ns)]
            protein = entry.find("xmlns:protein",self.ns)
            p_fullname = [ p.text for p in protein.findall("xmlns:recommendedName/xmlns:fullName", self.ns) ]
            p_altnames = [ p.text for p in protein.findall("xmlns:alternativeName/xmlns:fullName", self.ns) ]
            p_shortnames = [ p.text for p in protein.findall("xmlns:alternativeName/xmlns:shortName", self.ns) ]
            meta['protein_names'] = p_fullname + p_altnames + p_shortnames
            meta['organism'] = entry.find('xmlns:organism/xmlns:name[@type="scientific"]', self.ns).text
            metadata[name] = meta
            
        return metadata

    
class Pubtator:
    def __init__(self, database=None):
        """
        API docs: https://www.ncbi.nlm.nih.gov/research/pubtator/api.html
        RESTFull API: [Format]?[Type]=[Identifiers]&concepts=[Bioconcepts]
        """
        if database is not None:
            self.ner = pd.read_table(database, header=None, dtype=str)
            self.ner.columns = ['pmid','concept','conceptid','mentions', 'resource']
        self.url = "https://www.ncbi.nlm.nih.gov/research/pubtator-api/publications/export/"
        url = ""
        
    def submitPMIDList(self, pmids, concepts, fmt:str ='biocjson'):
        """
        pmids: 
            comma separated pmids.  pmids (for abstracts) or pmcids (for full-texts).
            Export our annotated publications in batches of up to 100 in GET or 1000 in POST, in BioC, pubtator or JSON formats.
        concepts: 
            Optional comma-delimited list of the bioconcept types to include in the results, 
            one or more of: gene, disease, chemical, species, mutation and/or cellline. 
            If this parameter is not present, then results will contain all six bioconcepts.
        fmt: 
            pubtator (pubtator), default
            biocxml (BioC-XML)
            biocjson (BioC-JSON):
        """
        url = self.url +fmt

        post = {}
        post = {"pmids": [pmid.strip() for pmid in pmids.split(",")]}

        #
        # load bioconcepts
        if concepts != "": 
            post["concepts"]=concepts.split(",")

        # request
        r = requests.post(url, json = post)
        if r.status_code != 200 :
            print ("[Error]: HTTP code "+ str(r.status_code))
            raise Exception("[Error]: HTTP code "+ str(r.status_code))
        
        if fmt == "biocjson":
            js = r.text.strip().split("\n")
            self.ret = [json.loads(j) for j in js]
            return self.ret
        self.ret = r.text.strip().split("\n")
        return self.ret
        #return r.json()
    def submitText(self):
        pass
    
    def getGeneMention(self):
        entities = {}
        for r in self.ret:
            pmid = r['id']
            res = []
            title = r['passages'][0] # titles
            abstract = r['passages'][1] # abstract
            offset = int(title['offset']) 
            for x in title['annotations']:
                #print("label: ", x['infons']['type'] )
                #print("text: ", x['text'])
                if x['infons']['type'].lower() == "gene":
                    res.append(x['text'])
                #start = int(x['locations'][0]['offset'])
                #length = int(x['locations'][0]['length'])
                #end = start+length - offset
                #print("location:",title['text'][start:end])

            offset = int(abstract['offset'])    
            for x in abstract['annotations']:
                if x['infons']['type'].lower() == "gene":
                    res.append(x['text'])
                #start = int(x['locations'][0]['offset']) - offset
                #length = int(x['locations'][0]['length'])
                #end = start+length
                #print("location:",abstract['text'][start:end])
                #  
            entities[pmid] = set(res)
        return entities
        #return set(res)
    
    
    def getGeneMentionAll(self, pmids=None):
        """
        This will return all gene entities from the full artical, regardless the distance to the mentions of MeSH
        """
        if not hasattr(self, 'ner'):
            raise Exception("Please load the Pubtator database first!")
        if isinstance(pmids, str):
            pmids =  [pmid.strip() for pmid in pmids.split(",")]
        ret = self.ner[self.ner.isin(pmids)]
        return ret
        
    
    def debug(self, ret):
        for r in ret:
            title = r['passages'][0] # titles
            abstract = r['passages'][1] # abstract
            offset = int(abstract['offset'])
            for x in title['annotations']:
                print("label: ", x['infons']['type'] )
                print("text: ", x['text'])
                start = int(x['locations'][0]['offset'])
                length = int(x['locations'][0]['length'])
                end = start+length
                print("location:",title['text'][start:end])
                #
                print("================")
                
            for x in abstract['annotations']:
                print("label: ", x['infons']['type'] )
                print("text: ", x['text'])
                start = int(x['locations'][0]['offset']) - offset
                length = int(x['locations'][0]['length'])
                end = start+length
                print("location:",abstract['text'][start:end])
                #
                print("================")
                
                
    
class GeneMeSHGraph(object):
    def __init__(self, gene_nodes: dict, mesh_nodes: dict, edges: list = None, nlp=None):
        """
        To build the edge between gene_node and mesh_node
        
        Mesh Nodes: dict, 
            weight: number of articals that has certain mesh    
            attribute: PMIDs, list of artical pmids that has the mesh term
        
        Gene Nodes: dict,
            weight: number of articals that has certain gene
            attribute: PMIDs, list of artical pmids that has the certain gene
        
        Edge: list of dict
            weight: number of co-occur articals
            attritbue:
                PMIDs. list of artical pmids that has mesh and gene co-occur
            nodes: undirected graph. 
            
        nlp: spay.load("bionlp_ner") or None
            
        """
        self.edges = edges
        self.gene_nodes = gene_nodes
        self.mesh_nodes = mesh_nodes
        self.nlp = nlp
    
    def _symbol2geneid(self):
        #symbol2id
        pass
        
    def _regex(self, text: str) -> list:
        # TODO
        pass
    
    def _pubtator_ner(self, pmid: list) -> list:
        """
        could process gene name recognition in batch
        Max limits ? 
        """
        self.pubtator.submitPMIDList(pmid, concepts='gene')
        pubtator_entities = self.pubtator.getGeneMention() # a list of set
        
    def _spacy_ner(self, text: str, nlp) -> set:
        """
        only process one artical at a time
        """
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            if ent.label_ == 'GENE_OR_GENE_PRODUCT':
                entities.append(ent.text)
        return set(entities)
    
    def _gene_node_index(self):
        """
        to reduce search space and time, index gene_nodes
        """
        symbol2geneid = {}
        for geneid, node in self.gene_nodes.items():
            symbol = node['gene_symbol']
            synonyms =  node['gene_synonyms']
            symbol2geneid[symbol] = geneid
        pass
    
    def _split_entities(self, gene_ners: set):
        """
        split gene entites into protenin names and gene names.
        """
        gn = []
        pn = []
        for g in gene_ners:
            g = g.strip()
            if g.find(" ") != -1:
                pn.append(g.lower()) # make protein names lower 
            if len(g) > 1: 
                gn.append(g) # need to be exact match
        return gn, pn
    
    @staticmethod
    def batch_pubtator(pubmed:dict):
        """
        pubmed: output from PubMedXMLParse.parse()
        """
        pubtator = Pubtator()            
        # get pubtator genes in batch to reduce query times
        pmid_batches = sorted(list(pubmed.keys()))
        gene_pubtator = {}
        gene_pubtator_batch = []
        strid = 1000
        for b in range(0, len(pmid_batches), strid):
            gene_pubtator = {}
            pmids = ",".join(pmid_batches[b:b+strid])
            ret = pubtator.submitPMIDList(pmids, concepts="gene", fmt="biocjson")
            g = pubtator.getGeneMention()
            gene_pubtator.update(g) ## slow
            #gene_pubtator_batch.append(g)
        #return gene_pubtator_batch    
        return gene_pubtator   
    
    def _name_hash(self):
        """
        """
        gene_index = {letter:set() for letter in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")+list("0123456789")}
        protein_index =  {letter:set() for letter in list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")+list("0123456789")}
        for geneid, node in self.gene_nodes.items():
            # concatenate all gene names
            gene_names = [node['gene_symbol']] + node['gene_synonyms']
            if 'gene_names' in node:
                # take care some genes don't have uniprot entry
                gene_names += node['gene_names']
            
            for g in gene_names:
                if g[0].upper() in gene_index:
                    gene_index[g[0].upper()].add(geneid)
                
            # protein name recognition  
            if 'protein_names' not in node: continue
            protein_names = np.unique([p.lower() for p in node['protein_names']])                
            for p in protein_names:
                if p[0].upper() in protein_index:
                    protein_index[p[0].upper()].add(geneid)
       
        return gene_index, protein_index
    def _concat(self,):
        # gene match
        for geneid, node in self.gene_nodes.items():          
            gene_names = [node['gene_symbol']] + node['gene_synonyms']
            if 'gene_names' in node:
                # take care some genes don't have uniprot entry
                gene_names += node['gene_names']
            # remove single letter alias 
            gene_names = np.unique([g for g in gene_names if len(g) > 2]) # at least 2

            self.gene_nodes[geneid]['_gene_merge'] = gene_names.tolist()             
            # protein name recognition  
            if 'protein_names' not in node: continue
            protein_names = np.unique([p.lower() for p in node['protein_names']])
            if len(protein_names) == 0: continue
            self.gene_nodes[geneid]['_protein_merge'] = protein_names.tolist() 
        
        
        
    def _exact_match(self, ners, pmid, gene_index, protein_index):
        """
        # TODO:
        ## Do we have a better search algorithm ??? 
        ## first letter to search, then 2nd letter ....
        ## in this way, we don't need to iterate the whole gene nodes
        """
        # gene_index, protein_index = self._name_hash()
        gene_ners, protein_ners = self._split_entities(ners)
        # gene_ners = np.array(list(gene_ners)) # set could not convert to list directly
        gene_ners = np.unique(gene_ners)
        protein_ners = np.unique(protein_ners) # already lower, just need to make unique 
        
        # gene match
        # TODO: if gene name or protein name belong to same geneid, it will be counted twice
        # we use set(), it won't be counted twice, we only record pmid. that's it
        for gene in gene_ners:
            gind = gene[0].upper()
            if gind not in gene_index: continue
            for geneid in gene_index[gind]:
                node = self.gene_nodes[geneid]
                if gene in node['_gene_merge']:
                    #breakpoint()
                    self.gene_nodes[geneid]['PMIDs'].add(pmid)
                    
        # protein name match
        for protein in protein_ners:
            pind = protein[0].upper()
            if pind not in protein_index: continue
            for geneid in protein_index[pind]:
                node = self.gene_nodes[geneid]
                if '_protein_merge' not in node: continue
                if protein in node['_protein_merge']:
                    self.gene_nodes[geneid]['PMIDs'].add(pmid)                                    
                
    def _exact_match_slow(self, ners, pmid):
        """
        Exact match. This is a time-limiting step. please use self._exact_match
        
        """
        gene_ners, protein_ners = self._split_entities(ners)
        # gene_ners = np.array(list(gene_ners)) # set could not convert to list directly
        gene_ners = np.unique([g for g in gene_ners if len(g) > 1])
        protein_ners = np.unique([p.lower() for p in protein_ners])
        ## Search the whole gene name database
        # It's case sensitive !
        
        ## split gene_nodes into A-Z partition, only need to store geneid        
        for geneid, node in self.gene_nodes.items():
            # concatenate all gene names
            gene_names = [node['gene_symbol']] + node['gene_synonyms']
            if 'gene_names' in node:
                # take care some genes don't have uniprot entry
                gene_names += node['gene_names']
            # remove single letter alias 
            gene_names = np.unique([g for g in gene_names if len(g) > 1])
            # search hits               
            mask = np.in1d(gene_names, gene_ners, assume_unique=True)

            if mask.sum() > 0:
                self.gene_nodes[geneid]['PMIDs'].add(pmid)
                continue # if matched, we could skip the protein name exact match

            # protein name recognition  
            if 'protein_names' not in node: continue
            protein_names = np.unique([p.lower() for p in node['protein_names']])
            if len(protein_names) == 0: continue
            mask = np.in1d(protein_names, protein_ners, assume_unique=True)
            if mask.sum() > 0:
                self.gene_nodes[geneid]['PMIDs'].add(pmid)   
                
                
    def gene_node_add_pmid(self, pubmed: dict, gene_pubtator=None):
        """
        Add pmids to gene_nodes
        
        pubmed: output from PubMedXMLParse.parse()
        """
        # gene name recognition
        # load bioNLP langeuage model from scispacy
        if self.nlp is None:
            self.nlp = spacy.load("en_ner_bionlp13cg_md")
        
        # init gene_nodes['PMIDs']
        for geneid, node in self.gene_nodes.items():
            self.gene_nodes[geneid]['PMIDs'] = set()
            
        # load pubtator
        if gene_pubtator is None:
            gene_pubtator = self.batch_pubtator(pubmed)
        # begin here     
        gene_index, protein_index = self._name_hash() # reduce search space of gene nodes
        # reduce computation time, concat genenames and protein names just once
        self._concat()
        

        for pmid, pub in pubmed.items(): 
            # if not research artical type, skip
            pubtype = pub['PublicationType'].values()
            if np.in1d(list(pubtype), ['Review', 'Editorial','Comment', 'Portrait', 'Published Erratum']).sum() > 0: 
                continue

            ## make sure that we each entry has title or abstracts.
            # if not, skip
            text = ""
            if ('Title' in pub) and (pub['Title']) is not None:
                text = pub['Title'] + "\n"
            if ('Abstract' in pub) and (pub['Abstract'] is not None):
                text +=  pub['Abstract']
            if len(text) < 5: 
                continue
            # spacy gene/protein recognition
            # _ners = self._spacy_ner(text, nlp)
            doc = self.nlp(text)     
            _ners = set(ent.text for ent in doc.ents if ent.label_ == 'GENE_OR_GENE_PRODUCT')
            # pubtator gene recognition
            #ret = pubtator.submitPMIDList(pmid, concepts="gene", fmt="biocjson")
            ## only 1 pmid each iteration, sometimes will cause http connection fail. use batch query
            #gene_pubtator = pubtator.getGeneMention()[pmid] # only 1 pmid each iteration

            # intersect the two
            if pmid in gene_pubtator:
                _ners = _ners.union(gene_pubtator[pmid])           
            if len(_ners) == 0: continue
            self._exact_match(_ners, pmid, gene_index, protein_index)
        
        # save space when nlp is done
        self.nlp = None

            
        
    def mesh_node_add_pmid(self, pubmed:dict):
        """
        Add pmids to mesh_nodes.
        
        pubmed: output from PubMedXMLParse.parse()
        """
        ## init mesh_node's pmid list record
        for desp_id, node in self.mesh_nodes.items():
            self.mesh_nodes[desp_id]['PMIDs'] = set()
        
        ## mesh_nodes will record pmids if the node is appeared in a artical
        # TODO: only consider research articals 
        for pmid, pub in pubmed.items():
            # if a pmid has no mesh terms, skip
            if 'MeSH' not in pub: 
                continue
            for mesh in pub['MeSH']:
                mid = mesh['DescriptorUI']
                self.mesh_nodes[mid]['PMIDs'].add(pmid)    
        
    def edge_add2(self):
        """
        build links between gene and mesh term.
        
        edge node: undirected graph. gene or mesh term
        edge weight: number of common pmids (> 0). if 0, not edge.
        edge PMIDs: common pmids
        """
        self.edges = []

        for meshid, mesh_node in self.mesh_nodes.items():
            mp = np.array(list(mesh_node['PMIDs']))
            for geneid, gene_node in self.gene_nodes.items():
                gp = np.array(list(gene_node['PMIDs']))
                edge = {'gene_node':geneid, 'mesh_node': meshid}     
                mask = np.in1d(gp, mp, assume_unique=True)
                edge['weight'] = mask.sum()
                if edge['weight'] > 0: 
                    edge['PMIDs'] = gp[mask].tolist()
                    self.edges.append(edge)
                    
    def edge_add(self):
        """
        build links between gene and mesh term.
        
        edge node: undirected graph. gene or mesh term
        edge weight: number of common pmids (> 0). if 0, not edge.
        edge PMIDs: common pmids
        """
        self.edges = []
        for geneid, gene_node in self.gene_nodes.items():
            for meshid, mesh_node in self.mesh_nodes.items():
                edge = {'gene_node':geneid, 'mesh_node': meshid}
                comm = gene_node['PMIDs'].intersection(mesh_node['PMIDs'])
                edge['weight'] = len(comm)
                edge['PMIDs'] = comm
                if edge['weight'] > 0:
                    self.edges.append(edge)                    
    def to_networkx(self):
        """
        return networkX formated Graph
        """
        
        G = nx.Graph()
        for gid, gnode in self.gene_nodes.items():
            if "_gene_merge" in gnode:
                del gnode['_gene_merge']
            if "_protein_merge" in gnode:
                del gnode['_protein_merge']
            # Add nodes with the node attribute "bipartite" for bipartite graph
            gnode['bipartite'] = 0
            gnode['weight'] = len(gnode['PMIDs'])
            G.add_node(gid, **gnode)

        for mid, mnode in self.mesh_nodes.items():
            mnode['bipartite'] = 1
            mnode['weight'] = len(mnode['PMIDs'])
            G.add_node(mid, **mnode)
        for e in self.edges:
            G.add_edge(e['gene_node'], e['mesh_node'], weight=e['weight'],PMIDs=e['PMIDs'])
        
        return G
