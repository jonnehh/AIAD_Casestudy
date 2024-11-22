import pandas as pd

def process_miss_data(data: pd.DataFrame) -> pd.DataFrame:
    # Handle missing values for Age, Embarked, and Fare
    data['Age'] = data['Age'].fillna(data.groupby(['Pclass', 'Sex'])['Age'].transform('median'))
    data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
    data['Fare'] = data['Fare'].fillna(data.groupby('Pclass')['Fare'].transform('median'))
    
    # Ensure no NaN values remain in other columns
    data['SibSp'].fillna(data['SibSp'].median(), inplace=True)
    data['Parch'].fillna(data['Parch'].median(), inplace=True)
    
    return data

def feature_engineering(data: pd.DataFrame) -> pd.DataFrame:
    # Create new features based on EDA steps
    data['HasCabin'] = data['Cabin'].notna().astype(int)
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['Title'] = data['Name'].apply(lambda x: x.split(',')[1].split('.')[0].strip()).map({
        "Mr": "Mr", "Miss": "Miss", "Mrs": "Mrs", "Master": "Master",
        "Dr": "Other", "Rev": "Other", "Col": "Other", "Mlle": "Miss",
        "Major": "Other", "Mme": "Mrs", "Capt": "Other", "Countess": "Other",
        "Jonkheer": "Other", "Don": "Other"
    })
    data['IsAlone'] = (data['FamilySize'] == 1).astype(int)
    data['FareBin'] = pd.qcut(data['Fare'], 4, labels=[1, 2, 3, 4])
    
    # Create Age bands
    data['Age_band'] = pd.cut(data['Age'], bins=[0, 16, 32, 48, 64, 80], labels=[0, 1, 2, 3, 4])
    
    # Encode categorical features using one-hot encoding
    data = pd.get_dummies(data, columns=['Sex', 'Embarked', 'Title'], drop_first=True)
    
    return data

def drop_unnecessary_features(data: pd.DataFrame) -> pd.DataFrame:
    # Drop columns that are not needed for modeling
    data.drop(['Name', 'Ticket', 'Cabin', 'PassengerId'], axis=1, inplace=True)
    return data

