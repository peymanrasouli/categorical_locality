import sys
sys.path.insert(1,"domain_knowledge/OWL2Vec-Star")
from owl2vec_star import owl2vec_star
from gensim.models import KeyedVectors

class GRAPH2EMBEDDING(object):

    def __init__(self, owl_file):

        # Parameters:
        # ontology_file
        # config_file
        # uri_doc
        # lit_doc
        # mix_doc
        gensim_model = owl2vec_star.extract_owl2vec_model(owl_file, "domain_knowledge/default.cfg", True, True, True)

        output_folder = "domain_knowledge/embeddings/"

        # Gensim format
        gensim_model.save(output_folder + "adult_embeddings")

        # Txt format
        gensim_model.wv.save_word2vec_format(output_folder + "adult_embeddings.txt", binary=False)

        # load embedding model and Word2Vec Keyed Vectors
        model = KeyedVectors.load(output_folder + "adult_embeddings", mmap='r')
        self.model = model
        self.word2vec = model.wv

    def FindSimilarInstances(self, instance, N=None):

        # Most similar entities: cosmul
        instances = self.word2vec.most_similar_cosmul(positive=[instance],topn=N)

        return instances



        # vector = wv['pizza']  # Get numpy vector of a word
        # print("Vector for 'pizza'")
        # print(vector)
        #
        # # cosine similarity
        # similarity = wv.similarity('pizza', 'http://www.co-ode.org/ontologies/pizza/pizza.owl#Pizza')
        # print(similarity)
        #
        # similarity = wv.similarity('http://www.co-ode.org/ontologies/pizza/pizza.owl#Margherita', 'margherita')
        # print(similarity)
        #
        # # Most similar cosine similarity
        # result = wv.most_similar(positive=['margherita', 'pizza'])
        # print(result)
        #
        # # Most similar entities: cosmul
        # result = wv.most_similar_cosmul(positive=['margherita'])
        # print(result)


def main():

    # path of the knowledge graph file
    owl_file = "ontologies/adult_ontology_instantiated.owl"

    # instantiating the GRAPH2EMBEDDING class
    graph2embedding = GRAPH2EMBEDDING(owl_file)

    # finding similar instances to a specific instance
    N = 10
    instance = "individual0"
    similar_instances = graph2embedding.FindSimilarInstances(instance, N)

    print(similar_instances)

if __name__ == '__main__':
    main()
