import streamlit as st
import pandas as pd
import json
from openai import OpenAI
from dotenv import dotenv_values
from pycaret.clustering import setup, create_model,save_model,assign_model, predict_model
import plotly.express as px
from qdrant_client import QdrantClient

env=dotenv_values(".env")

EMBEDDING_DIM=3072
DATA='welcome_survey_simple_v2.csv'

@st.cache_resource
def get_openai_client():
     return OpenAI(api_key=env["OPENAI_API_KEY"])

if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"]=env["OPENAI_API_KEY"]
    else:
        st.info("Dodaj swój klucz API OpenAi aby korzystać z aplikacji")
        st.session_state["openai_api_key"]=st.text_input("Klucz API", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()

@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
    url=env["QDRANT_URL"], 
    api_key=env["QDRANT_API_KEY"],
    check_compatibility=False
)

@st.cache_data
def get_all_participants():
    all_df=pd.read_csv(DATA, sep=';')
    all_df["fav_place"].fillna("Brak danych", inplace=True)
    return all_df

source_df=get_all_participants()
setup(source_df,session_id=7)

# Tworzenie modelu
@st.cache_data
def get_model():
    
   
    kmeans=create_model('kmeans')
    save_model(kmeans,'welcome_survey_pipeline',verbose=False)
    return kmeans


kmeans=get_model()
df_with_clusters=assign_model(kmeans)

# cluster descriptipn
@st.cache_data
def get_cluster_descrptions():
   cluster_descriptions={}
   for cluster_id in df_with_clusters['Cluster'].unique():
       cluster_df=df_with_clusters[df_with_clusters['Cluster']==cluster_id]
          
       summary=""
       for column in df_with_clusters:
            if column=='Cluster':
                continue
            
            value_counts=cluster_df[column].value_counts()
            value_counts_str=', '.join([f"{idx}: {cnt}" for idx, cnt in value_counts.items()])
            summary += f"{column} - {value_counts_str}\n"
            cluster_descriptions[cluster_id]=summary
           
        
   return cluster_descriptions

# tworzenie prompt'a
@st.cache_data
def promt_generation():
    prompt="Użylismy algorytmu klastrowania."
    cluster_description=get_cluster_descrptions()
    for cluster_id, description in cluster_description.items():
        prompt +=f"\n\nKlaster {cluster_id}:\n{description}"
    prompt+="""
        Wygeneruj najlepsze nazwy dla każdego z klastrów oraz ich opisy

        Użyj formatu JSON. Przykładowo:
        { 
            "Cluster 0":{
                "name": "Klaster 0",
                "description": "W tym klastrze znajdują się osoby, które..."
            },
            "Cluster 1":{
                "name": "Klaster 1",
                "description": "W tym klastrze znajdują się osoby, które..."
            },
        }
    """
    return prompt
prompt=promt_generation()

cluster_description=get_cluster_descrptions()
openai_client=OpenAI(api_key=st.session_state["openai_api_key"])
response=openai_client.chat.completions.create(
    model="gpt-4",
    temperature=0,
    messages=[
        {
            "role": "user",
            "content": prompt,
        }
    ],
)
result=response.choices[0].message.content.replace("'''json","").replace("'''","").strip()
cluster_names_and_descriptions=json.loads(result)
   # return cluster_names_and_descriptions
st.session_state["Ai_description"]=cluster_names_and_descriptions  

with st.sidebar:
    with st.form("Formularz",border=False):
        st.header("Powiedz coś nam o sobie")
        st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
        age=st.selectbox("Wiek",['<18','25-34','45-54','35-44','18-24','55-64','>=65'])
        edu_level=st.selectbox("Wykształcenie",["Podstawowe",'Średnie','Wyższe'])
        fav_animals=st.selectbox("Ulubione zwierzęta",['Brak ulubionych','Psy','Koty','Psy i Koty','Inne'])
        fav_place=st.selectbox("Ulubione miejsce",['Nad wodą','W lesie','W górach','Inne'])
        gender=st.radio("Płeć",['Mężczyzna','Kobieta'])
        submitted=st.form_submit_button("Wyślij", type="primary")

person_df=pd.DataFrame([
    {  
        'age':age,
        'edu_level':edu_level,
        'fav_animals':fav_animals,
        'fav_place':fav_place,
        'gender':gender,
    }
])

predicted_cluster_id=predict_model(kmeans, data=person_df)["Cluster"].values[0]
predicted_cluster_data=st.session_state['Ai_description'][predicted_cluster_id]

st.title("Wyślij wypełniony formularz a dowiesz się do jakiej grupy pasujesz")

if submitted:

    st.header(f"Najbliżej ci do grupy:  {predicted_cluster_data['name']}")
    st.markdown(predicted_cluster_data['description'])

    same_cluster_df=df_with_clusters[df_with_clusters["Cluster"]==predicted_cluster_id]

    @st.fragment()
    def selection_done():
        selection=st.selectbox("Wybierz dla której kolumny chcesz zobaczyć wykres",['Wiek','Wykształcenie','Ulubione miejsce','Ulubione zwierzę','Płeć'])
   
        if selection =='Wiek':
            sel_column='age'
        elif selection=='Wykształcenie':
            sel_column='edu_level'
        elif selection=='Ulubione miejsce':
             sel_column='fav_place'
        elif selection=='Ulubione zwierzę':
            sel_column='fav_animals'
        else :
             sel_column='gender'

        fig=px.pie(same_cluster_df, names=sel_column)
        fig.update_layout(
        title=f"Rozkład {selection} w grupie",
        xaxis_title=selection,
        yaxis_title="Liczba osób",
        )
        st.plotly_chart(fig)

    selection_done()