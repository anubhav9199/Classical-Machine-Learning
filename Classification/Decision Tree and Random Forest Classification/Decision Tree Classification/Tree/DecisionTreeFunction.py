# Importing Libraries
import numpy as np

"""
This File Contains all the function which is used by a tree for Decision
tree Classification and Decision Tree Regression. It basically works on a flow
chart which is as follows
(1) Check Data is Pure
(2) if pure directly create leaf for Data
(3) else get all possible potential splits for Data
(4) From all potential splits choose perfectly dividing split line in Data
(5) Split the Data
(6) splited Data is save to the leaf created
(7) Repeat the algorithm (in Recurssive way) for getting completely pure Data
"""

def check_purity(data):
    
    """
    
    data
    PARAMETERS
    ----------
    data : df.values or numpy array
    no **args
    
    This function is use to check whether data is pure or not means data 
    contains only one class or not. Function needs only a numpy array of the
    original DataFrame.
    
    Returns
    -------
    It returns a boolean for the data is pure or not
    if data is pure then TRUE. (Boolean_value)
    else data is not pure then FALSE. (Boolean_value)
    
    """
    label_column = data[:,-1]
    unique_classes = np.unique(label_column)

    if len(unique_classes) == 1:
        return True
    else:
        return False
    
def create_leaf(data, mltask):
    """

    data, mltask
    
    PARAMETERS
    ----------
    data : DataFrame.values or numpy array
    mltask : Regresion or Classification
    
    This funnction create leaf for the data to save when it gets pure
    Returns
    -------
    leaf : gives the space to store data in when get pure. (Float_value)
    """
    label_column = data[:,-1]
    
    # regression
    if mltask == 'regression':
        leaf = np.mean(label_column)
    
    # classification
    else:
        unique_classes,count_unique_class = np.unique(label_column,return_counts = True)
        index = count_unique_class.argmax()
        leaf = unique_classes[index]    
    return leaf

def get_potential_split(data):
    
    """
    
    data
    PARAMETERS
    ----------
    data = DataFrame.values or numpy array
    
    This function is use to get all the potential splits between the data
    points. when the data is not pure to make it pure, we use this function to
    the split between the functions
    it act like gini-index in the tree. this function finds all the spliting 
    point (Gini-Index) in the function.
    
    Returns
    -------
    potential_splits : It returns a dictionary of all the possible split at a 
    given feature in any DataFrame. (Dictonary)
    
    """
    potential_splits = {}
    _, n_columns = data.shape
    for column_index in range(n_columns-1):
        values = data[:,column_index]
        unique_values = np.unique(values)

    potential_splits[column_index] = unique_values
    return potential_splits

def split_data(data,split_column,split_value):
    
    """
    data, split_column, split_value
    PARAMETERS
    ----------
    data : DataFrame.values or numpy array
    split_column : best column for spliting (from determine best split function)
    split_value : best value for spliting (from determine best split function)
    
    This function is use to split the data into two equal halve. this function
    is use to purify the data using the best value (Gini-Index value) and 
    best column select from the potential splits.
    
    Returns
    -------
    data_below : the data which remain below the best potential value. (Numpy Array)
    data_above : the data which remain above the best potential value. (Numpy Array)
    
    """
    split_column_value = data[:,split_column]
    type_of_feature = FEATURE_TYPE[split_column]

    if type_of_feature == 'continuous':
        data_below = data[split_column_value <= split_value]
        data_above = data[split_column_value > split_value]
    else:
        data_below = data[split_column_value == split_value]
        data_above = data[split_column_value != split_value]
    return data_below,data_above

def calculate_mse(data):
    
    """
    
    data
    PARAMETER
    ---------
    data : DataFrame.values or numpy array
    
    This Function is use to calculate mean square error when doing the 
    regression task of ML Decision Tree Regresion.
    
    Returns
    -------
    MSE : mean Square Error done when the model is fitting and predicting the 
    value (Float_value)
    
    """
    actual_value = data[:,-1]
    if len(actual_value) == 0:
        mse = 0
    else:
        predictions = np.mean(actual_value)
        mse = np.mean((actual_value - predictions)**2)
    return mse

def calculate_entropy(data):
    
    """
    data
    PARAMETER
    ---------
    data : DataFrame.values or numpy array
    
    This Function is use to calculate entropy of feature when doing the 
    regression task of ML Decision Tree Classification.
    
    Returns
    -------
    entropy : Entropy of feature while fitting and classifing class of data. (Float_value)
    
    """
    label_column = data[:,-1]
    _,counts = np.unique(label_column,return_counts = True)
    probabilites = counts/counts.sum()
    entropy = sum(probabilites * -np.log2(probabilites))    
    return entropy

def calculate_overall_metric(data_below,data_above, metric_function):
    
    """
    
    data_below,data_above, metric_function
    PARAMETER
    ---------
    data_below : numpy array
    data_above : numpy array
    metric_function : Function Name (calculate_mse or calculate_entropy)
    
    This function is use for predicting overall metric (MSE or Entropy). It 
    depends on the metric function passed whether calculate_mse or 
    calculate_entropy.
    
    Returns
    -------
    overall_metric : Overall_MSE or Overall_Entropy (both Float_value)
    
    """
    
    n_data_points = len(data_below) + len(data_above)
    p_data_below = len(data_below)/n_data_points
    p_data_above = len(data_above)/n_data_points

    overall_metric = (p_data_below * metric_function(data_below) 
                      + p_data_above * metric_function(data_above))
    return overall_metric

def determine_best_split(data, potential_splits, mltask):
    
    """
    
    data, potential_splits, mltask
    PARAMETER
    ---------
    data : DataFrame.values or numpy array
    potential_splits : Dictionary
    mltask : Regression or Classification
    
    This function is use determine best split column (Feature) and best value 
    of the featre for the spliting to make the data pure. It works to find the
    split between the feature of the data. The value is also known as Gini-Index.
    
    Returns
    -------
    best_split_column : The Index of best feature to split. (Integer_value)
    best_split_value : The best value of the feature (Gini-Index). (Float_value)
    
    """

    first_iteration = True
    for column_index in potential_splits:
        for value in potential_splits[column_index]:
            data_below,data_above = split_data(data, column_index, value)
            
            if mltask == 'regression':
                current_overall_metric = calculate_overall_metric(data_below, data_above, metric_function = calculate_mse)
                
            # classification
            else:
                current_overall_metric = calculate_overall_metric(data_below, data_above, metric_function = calculate_entropy)
            
            
            if first_iteration or current_overall_metric <= best_overall_metric:
                first_iteration = False
                
                best_overall_metric = current_overall_metric
                best_split_column = column_index
                best_split_value = value
    
    
    return best_split_column,best_split_value

def determin_type_of_feature(df):
    
    """
    
    df
    PARAMETER
    ---------
    df : Pandas DataFrame
    
    This function is use to determine the type of feature present in the data.
    Either the data has Continuous Feature value Or the data has Catagorical 
    Feature.
    
    Returns
    -------
    Feature_types : list of all the feature telling either continuous or 
    catagorical (list)
    
    """
    
    feature_types = []
    n_unique_value_threshold = 15
    
    for column in df.columns:
        unique_values = df[column].unique()
        example_values = unique_values[0]
        
        if (isinstance(example_values, str)) or (len(unique_values) <= n_unique_value_threshold):
            feature_types.append("catagorical")
        else:
            feature_types.append("continuous")
    return feature_types

def DecisionTreeAlgorithm(df, mltask, counter = 0, min_samples = 2, max_depth = 5):
    
    """
    
    df, mltask, counter = 0, min_samples = 2, max_depth = 5
    
    PARAMETER
    ---------
    df : Pandas DataFrame
    mltask : Regression or Classification
    counter : number of count function reccure itself. by default (0)
    min_samples : number of min_samples of the dataframe. by default (2)
    max_depth : Maximum Depth of tree. by default(5)
    
    This function is use to run the Decision Tree Algorithm. The function can 
    do both the Machine Learning Task (i.e. Regression or Classification) 
    depends on the user call.
    
    Returns
    -------
    tree : a patten of list (list_of_answer) in a dictionary (key = question)
    
    **pattern : { "Question" : ["Yes_Answer","No_Answer or {Reccurcive Function}"] }
    
    """

    if counter == 0:
        global COLUMN_HEADERS, FEATURE_TYPE
        COLUMN_HEADERS = df.columns
        FEATURE_TYPE = determin_type_of_feature(df)
        data = df.values
    else:
        data = df
        
    if (check_purity(data)) or (len(data) < min_samples) or (counter == max_depth):
        leaf = create_leaf(data, mltask)
        return leaf
    
    else:
        counter += 1
        
        potential_splits = get_potential_split(data)
        split_column,split_value = determine_best_split(data, potential_splits, mltask)
        data_below,data_above = split_data(data,split_column,split_value)
        
        if (len(data_below) == 0) or (len(data_above) == 0):
            leaf = create_leaf(data, mltask)
            return leaf
        
        feature_name = COLUMN_HEADERS[split_column]
        type_of_feature = FEATURE_TYPE[split_column]
        if type_of_feature == 'continuous':
            question = '{} <= {}'.format(feature_name,split_value)
        else:
            question = '{} = {}'.format(feature_name,split_value)
        sub_tree = {question:[]}
        
        yes_answer = DecisionTreeAlgorithm(data_below, mltask, counter, min_samples, max_depth)
        no_answer = DecisionTreeAlgorithm(data_above, mltask, counter, min_samples, max_depth)
        
        if yes_answer == no_answer :
            sub_tree = yes_answer
        else :
            sub_tree[question].append(yes_answer)
            sub_tree[question].append(no_answer)
        
        return sub_tree
    
def predict_example(example,tree):
    
    """
    
    example,tree
    PARAMETER
    ---------
    example : testing DataFrame
    tree : Tree
    
    This function is use to check whether the data prediction is accurate or not.
    
    Returns
    -------
    answer : value of the checking answer predict == actual TRUE or FALSE. (Boolean)
    
    """
    
    question = list(tree.keys())[0]
    feature_name, comparison, value = question.split()

    if comparison == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else :
            answer = tree[question][1]

    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else :
            answer = tree[question][1]
            
    if not isinstance(answer,dict):
        return answer
    else:
        residual_tree = answer
        return predict_example(example,residual_tree)
    
def classify_example(example,tree):
    
    """
    example,tree
    PARAMETER
    ---------
    example : testing DataFrame
    tree : Tree
    
    This function is use to check whether the data classifing is accurate or not.
    
    Returns
    -------
    answer : value of the checking answer predict == actual TRUE or FALSE. (Boolean)
    
    
    """
    
    question = list(tree.keys())[0]
    feature_name, comparison, value = question.split()

    if comparison == "<=":
        if example[feature_name] <= float(value):
            answer = tree[question][0]
        else :
            answer = tree[question][1]

    else:
        if str(example[feature_name]) == value:
            answer = tree[question][0]
        else :
            answer = tree[question][1]
            
    if not isinstance(answer,dict):
        return answer
    else:
        residual_tree = answer
        return classify_example(example,residual_tree)
    
def calculate_accuracy(df,tree):
    
    """
    
    df,tree
    PARAMETER
    ---------
    df : testing DataFrame
    tree : Tree
    
    This Function is use to find the accuracy of the Classification model.
    
    Returns
    -------
    accuarcy : accuracy of the Model. (Float_value)
    
    """

    df['classification'] = df.apply(classify_example, axis =1, args = (tree,))
    df['classification_correct'] = df.classification == df.label
    
    accuracy = df.classification_correct.mean()
    
    return accuracy

def calculate_r_square(df,tree):
    
    """
    df,tree
    PARAMETER
    ---------
    df : testing DataFrame
    tree : Tree
    
    This Function is use to find the r square of the Regression model.
    
    Returns
    -------
    R Square : R Square of the Model. (Float_value)
    
    """
    
    label = df.label
    mean = label.mean()
    prediction = df.apply(predict_example, axis = 1, args = (tree,))
    
    ss_res = sum((label - prediction) ** 2)
    ss_tot = sum((label - mean) ** 2)
    r_squared = 1 - ss_res / ss_tot
    
    return r_squared
    