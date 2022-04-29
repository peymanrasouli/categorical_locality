from rdflib import Graph
from rdflib import URIRef
from rdflib import Namespace
from rdflib.namespace import RDF
import pandas as pd

class CSV2GRAPH(object):

    def __init__(self, rdf_file):

        # path to the RDF graph
        self.rdf_file = rdf_file

        #Note that this is the same namespace used in the ontology "adult_ontology.owl"
        self.adult_ns_str= "http://www.semanticweb.org/peyman/adult_ontology#"
        
        #Special namspaces class to create directly URIRefs in python.           
        self.adult = Namespace(self.adult_ns_str)

    def ParseKG(self):

        # Empty graph
        self.g = Graph()

        self.g.parse(self.rdf_file)
        print(len(self.g))

        # Prefixes for the serialization
        self.g.bind("adult", self.adult)

    def Convert(self, csv_file=None, data_frame=None):

        #Format:
        #      0                 1               2           3             4             5             6          7      8         9
        # AgeCategory   WeeklyWorkingHours   WorkClass   Education   MaritalStatus   Occupation   Relationship   Race   Sex   NativeCountry

        # read data to a pandas dataframe if CSV file is given
        if data_frame == None:
            # Load data in dataframe
            self.data_frame = pd.read_csv(csv_file, sep=',', quotechar='"', escapechar="\\")

        # parsing the adult ontology RDF graph
        self.ParseKG()

        ind = 0
        for row in self.data_frame.itertuples(index=False):

            # Mapping feature values (categories) to instances
            entity_individual_uri = self.adult_ns_str + 'individual'+str(ind)
            entity_agecategory_uri = self.adult_ns_str + row[0].lower()
            entity_weeklyworkinghours_uri = self.adult_ns_str + row[1].lower()
            entity_workclass_uri = self.adult_ns_str + row[2].lower()
            entity_education_uri = self.adult_ns_str + row[3].lower()
            entity_maritalstatus_uri = self.adult_ns_str + row[4].lower()
            entity_occupation_uri = self.adult_ns_str + row[5].lower()
            entity_relationship_uri = self.adult_ns_str + row[6].lower()
            entity_race_uri = self.adult_ns_str + row[7].lower()
            entity_sex_uri = self.adult_ns_str + row[8].lower()
            entity_nativecountry_uri = self.adult_ns_str + row[9].lower()

            #Types triples
            self.g.add((URIRef(entity_individual_uri), RDF.type, self.adult.Person))

            self.g.add((URIRef(entity_individual_uri), self.adult.hasAgeCategory, URIRef(entity_agecategory_uri)))
            self.g.add((URIRef(entity_individual_uri), self.adult.hasWeeklyWorkingHours, URIRef(entity_weeklyworkinghours_uri)))
            self.g.add((URIRef(entity_individual_uri), self.adult.hasWorkClass, URIRef(entity_workclass_uri)))
            self.g.add((URIRef(entity_individual_uri), self.adult.hasEducation, URIRef(entity_education_uri)))
            self.g.add((URIRef(entity_individual_uri), self.adult.hasMaritalStatus, URIRef(entity_maritalstatus_uri)))
            self.g.add((URIRef(entity_individual_uri), self.adult.hasOccupation, URIRef(entity_occupation_uri)))
            self.g.add((URIRef(entity_individual_uri), self.adult.hasRelationship, URIRef(entity_relationship_uri)))
            self.g.add((URIRef(entity_individual_uri), self.adult.hasRace, URIRef(entity_race_uri)))
            self.g.add((URIRef(entity_individual_uri), self.adult.hasSex, URIRef(entity_sex_uri)))
            self.g.add((URIRef(entity_individual_uri), self.adult.hasNativeCountry, URIRef(entity_nativecountry_uri)))

            ind += 1

    def saveGraph(self, file_output):
        self.g.serialize(destination=file_output, format='ttl')

def main():
    # CSV Format:
    # AgeCategory   WeeklyWorkingHours   WorkClass   Education   MaritalStatus   Occupation   Relationship   Race   Sex   NativeCountry

    # path of rdf and csv files
    rdf_file = "../domain_knowledge/adult_ontology.owl"
    csv_file = "../domain_knowledge/adult_categorical.csv"

    # instantiating the CSV2GRAPH class
    csv2graph = CSV2GRAPH(rdf_file)

    # Create RDF triples
    csv2graph.Convert(csv_file)

    # Graph with only data
    csv2graph.saveGraph("adult_ontology_filled.ttl")

if __name__ == '__main__':
    main()
