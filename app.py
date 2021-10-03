import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.neural_network import MLPClassifier

def main():
    from PIL import Image
    #image_hospital = Image.open('Neuro1.png')
    #image_ban = Image.open('Neuro2.png')
    #st.image(image_ban, use_column_width=False)
    #st.sidebar.image(image_hospital)
if __name__ == '__main__':
    main()


st.write("""
# Artificial neural network for predicting massive blood transfusion (for unseen data)

""")
st.write ("Tunthanathip et al.")

#st.write("""
### Performances of various algorithms from the training dataset [Link](https://pedtbi-train.herokuapp.com/)
#""")

#st.write ("""
### Labels of input features
#1.GCSer (Glasgow Coma Scale score at ER): range 3-15

#2.Hypotension (History of hypotension episode): 0=no , 1=yes

#3.pupilBE (pupillary light refelx at ER): 0=fixed both eyes, 1= fixed one eye, 2=React both eyes

#4.SAH (Subarachnoid hemorrhage on CT of the brain): 0=no, 1=yes

#""")


st.sidebar.header('User Input Features')

st.sidebar.markdown("""
[Example CSV input file](https://github.com/Thara-PSU/blood_transfusion/blob/main/example.csv)
""")

# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if  uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input_features():
        Disease = st.sidebar.slider('Disease (1.Tumor 2.Aneurysm 3.CVA 4.TBI 5.Spine-tumor 6.Spine-trauma 7.Spine-infect 8.Spine-degen 9.Cong-Brain 10.Cong-spine 11.Infection 12.Other)', 1, 12, 4)
        Operation = st.sidebar.slider('Operation (1.Cranio 2.DC 3.SOC/Retrosigmoid 4.Endoscope 5.Cranioplasty 6.Burr 7.Spine+inst 8.Spine-inst 9.Spine+cong 10.EVD 11.Shunt 12.Other)', 1, 12, 1)
        ASA_Class = st.sidebar.slider('ASA_class (Class 1-4)', 1, 4, 3)
        Warfarin = st.sidebar.slider('Warfarin use (0=no, 1=yes)', 0, 1, 0)
        Age = st.sidebar.slider('Age (year)', 0, 99, 55)
        Pre_Hb = st.sidebar.slider('Preop hemoglobin(g/dL)', 1, 20, 12)
        Pre_Hct = st.sidebar.slider('Preop hematocrit (%)', 10, 65, 37)
        Pre_PLT = st.sidebar.slider('Preop platelet count (x10^3/mcL)', 10, 550, 301)

        data = {'Disease': Disease,
                'Operation': Operation,
                'ASA_Class': ASA_Class,
                'Warfarin': Warfarin,
                'Age': Age,
                'Pre_Hb': Pre_Hb,
                'Pre_Hct': Pre_Hct,
                'Pre_PLT': Pre_PLT,
                }
        features = pd.DataFrame(data, index=[0])
        return features
    input_df = user_input_features()


# Combines user input features with entire penguins dataset
# This will be useful for the encoding phase
GBM_raw = pd.read_csv('train.2021.csv')
GBM = GBM_raw.drop(columns=['Massive_transfusion'])
df = pd.concat([input_df,GBM],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
encode = ['Disease','Pre_PLT']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df,dummy], axis=1)
    del df[col]
df = df[:1] # Selects only the first row (the user input data)

# Displays the user input features
st.subheader('User Input features')

if uploaded_file is not None:
    st.write(df)
else:
    st.write('Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).')
    st.write(df)


# Reads in saved classification model
load_clf = pickle.load(open('mass_ANN_clf.pkl', 'rb'))
 

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)

st.write("""# Prediction Probability""")
#st.subheader('Prediction Probability')
st.write(prediction_proba)

st.subheader('Class Labels and their corresponding index number')

label_name = np.array(['no','massive_transfusion'])
st.write(label_name)
# labels -dictionary
names ={0:'no',
1: 'massive_transfusion'}

st.write("""# Prediction""")
#st.subheader('Prediction')
two_year_survival = np.array(['no','massive_transfusion'])
st.write(two_year_survival[prediction])

#st.write("""# Prediction is high risk of massive blood transfusion when probability of the class 1 is more than 0.5""")

st.write ("""
### Other algorithms for predicting massive blood transfusion

""")


#st.markdown( "  [Random forest] (https://ct-pedtbi-test-rf.herokuapp.com/) ")
#st.markdown( "  [Logistic Regression] (https://ct-pedtbi-test-ln.herokuapp.com/) ")
#st.markdown( "  [Neural Network] (https://ct-pedtbi-test-nn.herokuapp.com/) ")
#st.markdown( "  [K-Nearest Neighbor (kNN)] (https://pedtbi-test-knn.herokuapp.com/) ")
#st.markdown( "  [naive Bayes] (https://ct-pedtbi-test-nb.herokuapp.com/) ")
#st.markdown( "  [Support Vector Machines ] (https://ct-pedtbi-test-svm.herokuapp.com/) ")
#st.markdown( "  [Gradient Boosting Classifier] (https://pedtbi-test-gbc.herokuapp.com/) ")
#st.markdown( "  [Nomogram] (https://psuneurosx.shinyapps.io/ct-pedtbi-nomogram/) ")

st.write ("""
### [Home](https://ct-pedtbi-home.herokuapp.com/)

""")
