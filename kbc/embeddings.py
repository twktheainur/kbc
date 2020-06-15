import torch
from torch import nn

from kbc.datasets import Dataset
from kbc.models import CP, KBCModel


class KnowledgeGraphEmbeddingExtractor:
    def __init__(self, dataset: Dataset, model: KBCModel, prefix: str = ""):
        self.dataset = dataset
        self.model = model
        self.prefix = prefix

    def _pref_uri(self, uri):
        return "<" + self.prefix + uri + ">"

    def global_entity_embedding(self, uri: str):
        pass

    def left_hand_side_entity_embedding(self, uri: str):
        id = self.dataset.get_node_id_from_name(self._pref_uri(uri))
        matrix = self.model.get_lhs(id, 1)
        return matrix.resize_(matrix.size()[0])

    def right_hand_side_entity_embedding(self, uri: str):
        id = self.dataset.get_node_id_from_name(self._pref_uri(uri))
        matrix = self.model.get_rhs(id, 1)
        return matrix.resize_(matrix.size()[0])

    def relation_embedding(self, uri: str):
        id = self.dataset.get_rel_id_from_name(self._pref_uri(uri))
        matrix = self.model.get_rel(id, 1)
        return matrix.resize_(matrix.size()[0])

    def similarity(self, uri_left, uri_rel, uri_right):
        lhs_vector = self.left_hand_side_entity_embedding(uri_left)
        rhs_vector = self.right_hand_side_entity_embedding(uri_right)
        rel_vector = self.relation_embedding(uri_rel)
        return torch.sum(lhs_vector * rel_vector * rhs_vector, 0, keepdim=True).tolist()[0]


dataset = Dataset("CKG-181019", use_cpu=True)
model = CP(dataset.get_shape(), 50)
model.load_state_dict(torch.load("models/CKG-181019.pickle", map_location=torch.device('cpu')))

kgee = KnowledgeGraphEmbeddingExtractor(dataset, model)

cos = nn.CosineSimilarity(dim=0)

topic_concepts = ["http://lod.gesis.org/thesoz/concept_10045504", "http://lod.gesis.org/thesoz/concept_10038824",
                  "http://lod.gesis.org/thesoz/concept_10035091",
                  "http://lod.gesis.org/thesoz/concept_10041774", "http://lod.gesis.org/thesoz/concept_10034501",
                  "http://vocabularies.unesco.org/thesaurus/concept407", "http://lod.gesis.org/thesoz/concept_10058252"]

example_claim = kgee.left_hand_side_entity_embedding(
    "http://data.gesis.org/claimskg/creative_work/087b8d48-f515-5265-b300-2235dbae76a2")

for topic in topic_concepts:
    print(
        topic + " -- " + str(kgee.similarity(
            "http://data.gesis.org/claimskg/creative_work/087b8d48-f515-5265-b300-2235dbae76a2",
            "http://purl.org/dc/terms/about",
            topic)))
pass

# Helathcare: http://lod.gesis.org/thesoz/concept_10045504
# Taxes: http://lod.gesis.org/thesoz/concept_10038824
# Education: http://lod.gesis.org/thesoz/concept_10035091
# Immigration: http://lod.gesis.org/thesoz/concept_10041774
# Elections: 	http://lod.gesis.org/thesoz/concept_10034501
# Crime: http://vocabularies.unesco.org/thesaurus/concept407
# Environment: http://lod.gesis.org/thesoz/concept_10058252
