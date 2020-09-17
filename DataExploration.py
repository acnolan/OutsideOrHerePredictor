import pandas as pd

def main():
    df = preprocessdata()
    tree = createDecisionTree(df)
    predictFromTree(df, tree)

def preprocessdata():
    #load the training data into a pandas dataframe
    csv_file = "rawdata.csv"

    df = pd.read_csv(csv_file, header=0)

    headings = list(df.columns.values)

    # Fill in missing previous word value
    df["Previous Word"].fillna("N/A", inplace = True)

    # Import label encoder and encode the categorical non-numeric features 
    from sklearn import preprocessing
    le = preprocessing.LabelEncoder()
    le.fit(df['Here or Outside'])
    df['Here or Outside'] = le.transform(df['Here or Outside'])

    le.fit(df['Previous Word'])
    df['Previous Word'] =le.transform(df['Previous Word'])

    # I am manually encoding the months because I believe label encoder may change when new data is added (not 100% sure but I don't want to risk it)
    encodedMonths = []
    for m in df.Month:
        if m == "January":
            encodedMonths.append(1)
        elif m == "February":
            encodedMonths.append(2)
        elif m == "March":
            encodedMonths.append(3)
        elif m == "April":
            encodedMonths.append(4)
        elif m == "May":
            encodedMonths.append(5)
        elif m == "June":
            encodedMonths.append(6)
        elif m == "July":
            encodedMonths.append(7)
        elif m == "August":
            encodedMonths.append(8)
        elif m == "September":
            encodedMonths.append(9)
        elif m == "October":
            encodedMonths.append(10)
        elif m == "November":
            encodedMonths.append(11)
        else:
            encodedMonths.appent(12)

    df.Month = encodedMonths

    # Same thing for time of day
    encodedtods = []
    for t in df["Time of day"]:
        if t == "Late Night":
            encodedtods.append(4) #"Late Night"
        elif t == "Morning":
            encodedtods.append(0) #"Morning"
        elif t == "Afternoon":
            encodedtods.append(1) #"Afternoon"
        elif t == "Evening":
            encodedtods.append(2) #"Evening"
        else:
            encodedtods.append(3) #"Night"

    df["Time of day"] = encodedtods
    print(df["Time of day"])  

    # For now we can drop the non numeric data and time values because date is unique and time is grouped in Time of Day
    df = df._get_numeric_data()

    print(df)
    print(df.describe())
    return df

def createDecisionTree(df):
    from sklearn import tree, model_selection

    # Define the target column, and the training features
    target = df["Here or Outside"].values
    feature_names = ["Day of week","Month","Previous Word","Time of day"]
    features = df[feature_names].values

    # We will try a lot of different values to determine the best feature for the decision tree. These variables will store those
    splitCriterion = "gini"
    maxDepth = 0
    minSampleLeaf = 0
    bestCrossVal = 0
    bestValue = 0

    # Loop to determine best split criterion, max-depth and min_samples_leaf
    for i in range(1,20):
        for j in range(1,20):
            for k in range(2):
                if k%2 == 0:
                    # Create a decision tree
                    decisionTree = tree.DecisionTreeClassifier(random_state=1, max_depth = i, min_samples_leaf = j, criterion="gini")

                    # do cross validation on our decision tree
                    for size in range(5,10):
                        X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, test_size=size/10, random_state=0)

                        # Fit the model
                        decisionTree_ = decisionTree.fit(X_train, y_train)

                        #capture the best cross validation test accuracy
                        if decisionTree_.score(X_test, y_test) > bestValue:
                            splitCriterion = "gini"
                            maxDepth = i
                            minSampleLeaf = j
                            bestCrossVal = size/10
                            bestValue = decisionTree_.score(X_test, y_test)
                else:
                    # Create a decision tree
                    decisionTree = tree.DecisionTreeClassifier(random_state=1, max_depth = i, min_samples_leaf = j, criterion="entropy")

                    # do cross validation on our decision tree
                    for size in range(5,10):
                        X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, test_size=size/10, random_state=0)

                        # Fit the model
                        decisionTree_ = decisionTree.fit(X_train, y_train)

                        #capture the best cross validation test accuracy
                        if decisionTree_.score(X_test, y_test) > bestValue:
                            splitCriterion = "entropy"
                            maxDepth = i
                            minSampleLeaf = j
                            bestCrossVal = size/10
                            bestValue = decisionTree_.score(X_test, y_test)

    # Create the decision tree with the best values
    print("The best max_depth is: ", maxDepth)
    print("The best min_samples_leaf is: ", minSampleLeaf)
    print("The best split criterion is: ", splitCriterion)
    print("The best percentage of data to train on for cross validation is: ", bestCrossVal)
    decisionTree = tree.DecisionTreeClassifier(random_state=1, max_depth = maxDepth, min_samples_leaf = minSampleLeaf, criterion=splitCriterion)             

    # Split the training data in the optimal way for cross validation test accuracy
    X_train, X_test, y_train, y_test = model_selection.train_test_split(features, target, test_size=bestCrossVal, random_state=0)

    # Fit the model with the best values
    decisionTree_ = decisionTree.fit(X_train, y_train)

    # Print the score on the training data
    print("Decision tree training data accuracy:")
    print(decisionTree_.score(X_train, y_train))

    # Print the score on the cross validation test data
    print("Decision tree cross validation accuracy:")
    print(decisionTree_.score(X_test, y_test))

    return decisionTree_

def predictFromTree(df, tree):
    # get day of the week
    import datetime
    dow = datetime.datetime.today().weekday() + 2
    if dow > 7:
        dow = 1

    # get time of day
    hour = datetime.datetime.now().hour
    tod = 3 #"Night"
    if hour < 4:
        tod = 4 #"Late Night"
    elif hour < 12:
        tod = 0 #"Morning"
    elif hour < 16:
        tod = 1 #"Afternoon"
    elif hour < 18:
        tod = 2 #"Evening"

    # get most recent answer
    previousWord = df["Here or Outside"].iloc[-1]

    # get month
    month = datetime.datetime.now().month

    # make the vector to predict
    test_features = [[dow, month, previousWord, tod]]
    print(test_features)

    predictions = tree.predict(test_features)
    print(predictions)


    return


if __name__ == '__main__':
    main()