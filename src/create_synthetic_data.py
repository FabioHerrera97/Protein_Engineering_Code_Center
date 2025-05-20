import pandas as pd
import numpy as np
import random
from typing import List, Dict, Optional, Union

class ProteinVariantGenerator:

    '''
    A generator class for simulating protein single-point variants with associated structural 
    and functional annotations such as position, distance to active site, secondary structure, 
    and log-fitness values.

    This class is useful for generating synthetic datasets to test machine learning models 
    for protein engineering tasks or benchmarking algorithms for variant effect prediction.

    Parameters:
    ----------
    n_samples : int, default=950
        Number of variants to generate when calling `generate_data_frame()`.
    
    seed : int, optional, default=42
        Seed for random number generation to ensure reproducibility.
    
    amino_acids : List[str], optional
        List of amino acids to sample from. Defaults to the 20 standard amino acids.
    
    secondary_structures : List[str], optional
        List of secondary structure labels. Defaults to ['alpha', 'beta', 'turn', 'coil', 'loop'].
    
    position_range : tuple, default=(1, 500)
        Tuple specifying the inclusive range of positions for mutations.
    
    distance_range : tuple, default=(1, 20)
        Tuple specifying the range of distances (in Ångströms) from the active site.
    
    positive_fitness_prob : float, default=0.3
        Probability that a generated variant will have a positive log-fitness score.
    
    positive_fitness_params : dict, optional
        Parameters for sampling positive log-fitness values. Should include 'loc', 'scale', and 'max'.
        Defaults to {'loc': 1, 'scale': 1, 'max': 3}.
    
    negative_fitness_params : dict, optional
        Parameters for sampling negative log-fitness values. Should include 'loc', 'scale', and 'min'.
        Defaults to {'loc': 3, 'scale': 2, 'min': -7}.

    Methods:
    -------
    generate_variant_id(position: int) -> str
        Generates a mutation identifier string in the format "X123Y".
    
    generate_position() -> int
        Randomly samples a mutation position within the specified range.
    
    generate_distance() -> float
        Randomly samples a distance from the active site within the specified range.
    
    generate_secondary_structure() -> str
        Randomly selects a secondary structure label.
    
    generate_log_fitness() -> float
        Samples a log-fitness value, either positive or negative, according to specified distributions.
    
    generate_single_variant() -> Dict
        Generates a dictionary representing a single protein variant with all associated attributes.
    
    generate_data_frame(n_samples: Optional[int] = None) -> pd.DataFrame
        Generates a DataFrame containing synthetic data for multiple variants.
    
    set_parameters(...)
        Updates internal parameters such as number of samples, amino acid list, structural features,
        and fitness distributions.
    '''
    def __init__(self, 
                 n_samples: int = 950,
                 seed: Optional[int] = 42,
                 amino_acids: Optional[List[str]] = None,
                 secondary_structures: Optional[List[str]] = None,
                 position_range: tuple = (1, 500),
                 distance_range: tuple = (1, 20),
                 positive_fitness_prob: float = 0.3,
                 positive_fitness_params: Optional[Dict[str, Union[float, int]]] = None,
                 negative_fitness_params: Optional[Dict[str, Union[float, int]]] = None):
        
        self.n_samples = n_samples
        self.seed = seed
        self.positive_fitness_prob = positive_fitness_prob
        self._set_random_seeds()
        
        self.amino_acids = amino_acids or ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                                         'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self.secondary_structures = secondary_structures or ['alpha', 'beta', 'turn', 'coil', 'loop']
        
        self.position_range = position_range
        self.distance_range = distance_range
        
        self.positive_fitness_params = positive_fitness_params or {'loc': 1, 'scale': 1, 'max': 3}
        self.negative_fitness_params = negative_fitness_params or {'loc': 3, 'scale': 2, 'min': -7}
    
    def _set_random_seeds(self):

        np.random.seed(self.seed)
        random.seed(self.seed)
    
    def generate_variant_id(self, position: int) -> str:

        orig_aa = random.choice(self.amino_acids)
        mut_aa = random.choice([aa for aa in self.amino_acids if aa != orig_aa])
        variant_id = f'{orig_aa}{position}{mut_aa}'
        return variant_id
    
    def generate_position(self) -> int:
        
        position = random.randint(*self.position_range)
        return position
    
    def generate_distance(self) -> float:
        
        distance = round(random.uniform(*self.distance_range), 2)
        return distance
    
    def generate_secondary_structure(self) -> str:
        
        secondary_structure = random.choice(self.secondary_structures)
        return secondary_structure
    
    def generate_log_fitness(self) -> float:
        
        if random.random() < self.positive_fitness_prob:
            val = abs(np.random.normal(
                loc=self.positive_fitness_params['loc'],
                scale=self.positive_fitness_params['scale']
            ))
            val = min(val, self.positive_fitness_params['max'])
        else:
            val = -abs(np.random.normal(
                loc=self.negative_fitness_params['loc'],
                scale=self.negative_fitness_params['scale']
            ))
            val = max(val, self.negative_fitness_params['min'])
            log_fitness = round(val, 3)
        return log_fitness
    
    def generate_single_variant(self) -> Dict:

        position = self.generate_position()
        single_variant = {
            'ID': self.generate_variant_id(position),
            'Position': position,
            'Distance_AS': self.generate_distance(),
            'Secondary_structure': self.generate_secondary_structure(),
            'Log_fitness': self.generate_log_fitness()
        }
        return single_variant
    
    def generate_data_frame(self, n_samples: Optional[int] = None) -> pd.DataFrame:
  
        n = n_samples if n_samples is not None else self.n_samples
        data = [self.generate_single_variant() for _ in range(n)]
        df_data = pd.DataFrame(data)
        return df_data
    
    def set_parameters(self, 
                      n_samples: Optional[int] = None,
                      amino_acids: Optional[List[str]] = None,
                      secondary_structures: Optional[List[str]] = None,
                      position_range: Optional[tuple] = None,
                      distance_range: Optional[tuple] = None,
                      positive_fitness_prob: Optional[float] = None,
                      positive_fitness_params: Optional[dict] = None,
                      negative_fitness_params: Optional[dict] = None):

        if n_samples is not None:
            self.n_samples = n_samples
        if amino_acids is not None:
            self.amino_acids = amino_acids
        if secondary_structures is not None:
            self.secondary_structures = secondary_structures
        if position_range is not None:
            self.position_range = position_range
        if distance_range is not None:
            self.distance_range = distance_range
        if positive_fitness_prob is not None:
            self.positive_fitness_prob = positive_fitness_prob
        if positive_fitness_params is not None:
            self.positive_fitness_params.update(positive_fitness_params)
        if negative_fitness_params is not None:
            self.negative_fitness_params.update(negative_fitness_params)