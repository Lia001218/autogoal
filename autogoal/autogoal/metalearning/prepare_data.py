import numpy as np
from odmantic import SyncEngine
from autogoal.database.metafeature_model import MetafeatureModel



def create_db_from_tabular_metafeatures():
    db = SyncEngine(database= 'Metalearning')
    tabular_features_db = []
    tabular_modality = db.find(MetafeatureModel, MetafeatureModel.dataset_type == 'TabularMetafeatureExtractor')
    for i in tabular_modality:
        if i.pipelines:
            for j in i.pipelines:
                current= [i.metacaracteristic_model,i.metric,j.algorithm_flow,j.eval_result]
                tabular_features_db.append(current)
    return np.array(tabular_features_db)

def create_db_from_text_metafeatures():
    text_features_db = np.array([])
    text_modality = db.find(MetafeatureModel, MetafeatureModel.dataset_type == 'TextMetafeatureExtractor')
    for i in text_modality:
        if i.pipelines:
            for j in i.pipelines:
                current= [i.metacaracteristic_model,i.metric,j.algorithm_flow,j.eval_result]
                text_features_db = np.append(text_features_db,current)
    return np.array(text_features_db)

def create_db_from_image_metafeatures():
    image_features_db = np.array([])
    image_modality = db.find(MetafeatureModel, MetafeatureModel.dataset_type == 'ImageMetafeatureExtractor')
    for i in image_modality:
        if i.pipelines:
            for j in i.pipelines:
                current= [i.metacaracteristic_model,i.metric,j.algorithm_flow,j.eval_result]
                image_features_db = np.append(image_features_db,current)
    return np.array(image_features_db)
        
