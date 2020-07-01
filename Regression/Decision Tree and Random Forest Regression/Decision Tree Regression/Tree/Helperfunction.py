# Importing Library
import random

def train_test_split(df,test_size):
    
    """
    df, test_size
    
    PARAMETER
    ---------
    df : Pandas DataFrame
    test_size : test_size (Integer or Float) value
    
    This Function is  use to divide the data into train and test DataFrame.
    
    Returns
    -------
    train_df : Training DataFrame. (Pandas DataFrame)
    test_df : Testing DataFrame. (Pandas DataFrame)
    
    """
    if isinstance(test_size,float):
        test_size = round(test_size * len(df))
    index = df.index.tolist()
    test_index = random.sample(population=index,k=test_size)
    test_df = df.loc[test_index]
    train_df = df.drop(test_index)    
    return train_df,test_df
