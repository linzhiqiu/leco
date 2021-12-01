# To setup semi-inat: First download all the data in https://github.com/cvl-umass/semi-inat-2021
import json
# annotation_file = ./semi_inat/annotation_v2.json

from datasets import SemiInat2021

# class SemiInat2021(LecoDataset):
#     def __init__(self, data_dir):
#         super().__init__(data_dir)
    
#     def get_dataset(self):
#         normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         transform_train = transforms.Compose(
#             [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize]
#         )
#         transform_test = transforms.Compose(
#             [transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(), normalize]
#         )

#         trainset = IndexFolder(
#             os.path.join(self.data_dir, 'semi_inat'),
#             transform=transform_train
#         )
#         testset = IndexFolder(
#             os.path.join(self.data_dir, 'semi_inat'),
#             transform=transform_test
#         )
#         return trainset, testset
    
#     def get_class_hierarchy(self):
#         # An example of taxa
#         # {'class_id': 10, 'species': 'Viola selkirkii', 'kingdom': 'Plantae', 'phylum': 'Tracheophyta', 'class': 'Magnoliopsida', 'order': 'Malpighiales', 'family': 'Violaceae', 'genus': 'Viola'}
#         hierachy = {}
#         annotation_file = os.path.join(self.data_dir, 'semi_inat', 'annotation_v2.json')
#         if not os.path.exists(annotation_file):
#             raise Exception(f"File {annotation_file} does not exist.")
#         annotation_category = json.load(open(annotation_file, 'r'))
        
#         for taxa in annotation_category['categories']:
#             if not taxa['kingdom'] in hierachy:
#                 hierachy[taxa['kingdom']] = {}
#             if not taxa['phylum'] in hierachy[taxa['kingdom']]:
#                 hierachy[taxa['kingdom']][taxa['phylum']] = {}
#             if not taxa['class'] in hierachy[taxa['kingdom']][taxa['phylum']]:
#                 hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']] = {}
#             if not taxa['order'] in hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']]:
#                 hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']] = {}
#             if not taxa['family'] in hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']]:
#                 hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']] = {}
#             if not taxa['genus'] in hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']]:
#                 hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']][taxa['genus']] = []
#             hierachy[taxa['kingdom']][taxa['phylum']][taxa['class']][taxa['order']][taxa['family']][taxa['genus']].append(taxa)
        
#         num_kingdom = 0
#         num_phylum = 0
#         num_class = 0
#         num_order = 0
#         num_family = 0
#         num_genus = 0
#         num_species = 0
        
#         all_taxa = {
#             'kingdom' : [],
#             'phylum' : [],
#             'class' : [],
#             'order' : [],
#             'family' : [],
#             'genus' : [],
#             'species' : []
#         }
#         for kingdom in hierachy:
#             num_kingdom += 1
#             all_taxa['kingdom'].append(kingdom)
#             print(f"{kingdom} kingdom has {len(hierachy[kingdom])} phylum")
#             for phylum in hierachy[kingdom]:
#                 num_phylum += 1
#                 all_taxa['phylum'].append(phylum)
#                 print(f"\t{phylum} phylum has {len(hierachy[kingdom][phylum])} classes")
#                 for the_class in hierachy[kingdom][phylum]:
#                     num_class += 1
#                     all_taxa['class'].append(the_class)
#                     print(f"\t\t{the_class} class has {len(hierachy[kingdom][phylum][the_class])} orders")
#                     for order in hierachy[kingdom][phylum][the_class]:
#                         num_order += 1
#                         all_taxa['order'].append(order)
#                         print(f"\t\t\t{order} order has {len(hierachy[kingdom][phylum][the_class][order])} family")
#                         for family in hierachy[kingdom][phylum][the_class][order]:
#                             num_family += 1
#                             all_taxa['family'].append(family)
#                             print(f"\t\t\t\t{family} family has {len(hierachy[kingdom][phylum][the_class][order][family])} genus")
#                             for genus in hierachy[kingdom][phylum][the_class][order][family]:
#                                 num_genus += 1
#                                 all_taxa['genus'].append(genus)
#                                 print(f"\t\t\t\t\t{genus} genus has {len(hierachy[kingdom][phylum][the_class][order][family][genus])} species")
#                                 for species in hierachy[kingdom][phylum][the_class][order][family][genus]:
#                                     num_species += 1
#                                     all_taxa['species'].append(species)
#                                     pass
#         print()
#         print(f'Number of kingdom: {num_kingdom}')
#         print(f'Number of phylum: {num_phylum}')
#         print(f'Number of class: {num_class}')
#         print(f'Number of order: {num_order}')
#         print(f'Number of family: {num_family}')
#         print(f'Number of genus: {num_genus}')
#         print(f'Number of species: {num_species}')

#         tp_names = ['kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'species']
#         leaf_idx_to_all_class_idx = {} # leaf_idx_to_all_class_idx[leaf_idx][super_class_time] = super_class_idx at super_class_time of this leaf class 
#         all_tp_info = []
#         for tp_idx in range(len(tp_names)-1, -1, -1):
#             tp_info = {
#                 'tp_name' : tp_names[tp_idx],
#                 'tp_idx' : tp_idx,
#                 'all_classes' : [],
#                 'num_of_classes' : 0,
#                 'idx_to_leaf_name' : {},
#                 'leaf_name_to_idx' : {},
#             }
#             if tp_idx == len(tp_names)-1:
#                 tp_name = tp_names[tp_idx]
#                 assert tp_name == 'species'
#                 for taxa in all_taxa['species']:
#                     class_id = taxa['class_id']
#                     species = taxa['species']
#                     tp_info['idx_to_leaf_name'][class_id] = species
#                     tp_info['leaf_name_to_idx'][species] = class_id
#                     # genus = taxa['genus']
#                     # family = taxa['family']
#                     # order = taxa['order']
#                     # the_class = taxa['class']
#                     # phylum = taxa['phylum']
#                     # kingdom = taxa['kingdom']
#                     leaf_idx_to_all_class_idx[class_id] = [
#                         all_taxa[level].index(taxa[level])
#                         for level in tp_names[:-1]
#                     ] + [class_id]
                
#                 for class_id in sorted(list(leaf_idx_to_all_class_idx.keys())):
#                     tp_info['all_classes'].append(tp_info['idx_to_leaf_name'][class_id])
#             else:
#                 tp_info['all_classes'] = all_taxa[tp_names[tp_idx]]
#                 for class_id, leaf_name in enumerate(tp_info['all_classes']):
#                     tp_info['idx_to_leaf_name'][class_id] = leaf_name
#                     tp_info['leaf_name_to_idx'][leaf_name] = class_id
#             tp_info['num_of_classes'] = len(tp_info['all_classes'])

#             all_tp_info = [tp_info] + all_tp_info

#         # leaf_idx_to_all_class_idx = {} # leaf_idx_to_all_class_idx[leaf_idx][super_class_time] = super_class_idx at super_class_time of this leaf class 
#         for leaf_idx in leaf_idx_to_all_class_idx:
#             if not leaf_idx_to_all_class_idx[leaf_idx][-1] == leaf_idx:
#                 print("Wrong label format: Last index must be leaf index")
#                 import pdb; pdb.set_trace()
#         for i, tp_info in enumerate(all_tp_info):
#             assert tp_info['tp_idx'] == i
#         return all_tp_info, leaf_idx_to_all_class_idx

if __name__ == '__main__':
    dataset = SemiInat2021('/ssd1/leco/')
    trainset, testset = dataset.get_dataset()

    all_tp_info, leaf_idx_to_all_class_idx = dataset.get_class_hierarchy()
    
    import pdb; pdb.set_trace()
    for tp_info in all_tp_info:
        print(f"Level ({tp_info['tp_name']})")
        tp_idx = tp_info['tp_idx']
        for name, sets in [['Train', trainset], ['Test', testset]]:
            print(f"\t{name}:")
            counter = {leaf_name: 0 for leaf_name in tp_info['leaf_name_to_idx']}
            for _, label in sets:
                leaf_idx = leaf_idx_to_all_class_idx[label][tp_idx]
                leaf_name = tp_info['idx_to_leaf_name'][leaf_idx]
                counter[leaf_name] += 1
            for leaf_name in counter:
                print(f"\t\t{leaf_name}: {counter[leaf_name]}")
                
            
    import pdb; pdb.set_trace()