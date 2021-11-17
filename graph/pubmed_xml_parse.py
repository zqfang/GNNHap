import sys
import gzip
import pandas as pd
import xml.etree.ElementTree as ET 
import joblib

class PubMedXMLPaser:
    def __init__(self, xml):
        if xml.endswith(".gz"):
            xml_handle = gzip.open(xml,'r')
        else:
            xml_handle = xml

        xtree = ET.parse(xml_handle)
        self.xroot = xtree.getroot()
        if xml.endswith(".gz"):
            xml_handle.close() 

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
            art['Language'] = particle.find("./MedlineCitation/Article/Language").text ## do we need to worry about this if only the abstract is needed
            year = particle.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Year")
            if year is not None:
                art['Year'] = year
            month = particle.find("./MedlineCitation/Article/Journal/JournalIssue/PubDate/Month")
            if month is not None:
                art['Month'] = month
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
    
if __name__ == "__main__":
    
    xml = sys.argv[1] # path to xml
    output = sys.argv[2]    
    pubmed = PubMedXMLPaser(xml)
    meta = pubmed.parse()
    joblib.dump(meta, output)
