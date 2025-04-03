import numpy as np

class SimplifiedBaggingRegressor:
    def __init__(self, num_bags, oob=False):
        self.num_bags = num_bags
        self.oob = oob
        
    def _generate_splits(self, data: np.ndarray):
        self.indices_list = []
        data_length = len(data)
        for _ in range(self.num_bags):
            bootstrap_indices = np.random.choice(data_length, data_length, replace=True)
            self.indices_list.append(bootstrap_indices)
        
    def fit(self, model_constructor, data, target):
        self.data = None
        self.target = None
        self._generate_splits(data)

        
        self.models_list = []
        for bootstrap_indices in self.indices_list:
            model = model_constructor()
            data_bag = data[bootstrap_indices]
            target_bag = target[bootstrap_indices]
            self.models_list.append(model.fit(data_bag, target_bag))
        
        if self.oob:
            self.data = data
            self.target = target
        
    def predict(self, data):

        predictions = np.array([model.predict(data) for model in self.models_list])
        return predictions.mean(axis=0)
    
    def _get_oob_predictions_from_every_model(self):

        list_of_predictions_lists = [[] for _ in range(len(self.data))]
        
        for model, bootstrap_indices in zip(self.models_list, self.indices_list):
            oob_mask = np.ones(len(self.data), dtype=bool)
            oob_mask[bootstrap_indices] = False
            oob_data = self.data[oob_mask]
            oob_indices = np.where(oob_mask)[0]
            
            if len(oob_data) > 0:
                oob_predictions = model.predict(oob_data)
                for index, pred in zip(oob_indices, oob_predictions):
                    list_of_predictions_lists[index].append(pred)
        
        self.list_of_predictions_lists = np.array(list_of_predictions_lists, dtype=object)
    
    def _get_averaged_oob_predictions(self):

        self._get_oob_predictions_from_every_model()
        self.oob_predictions = np.array([
            np.mean(predictions) if len(predictions) > 0 else np.nan
            for predictions in self.list_of_predictions_lists
        ])

        
    def OOB_score(self):

        self._get_averaged_oob_predictions()
        valid_predictions_mask = ~np.isnan(self.oob_predictions)
        mse = np.mean((self.target[valid_predictions_mask] - self.oob_predictions[valid_predictions_mask])**2)
        return mse
