import sys
import streamlit as st
import pdfplumber
import gensim
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
from gensim.models.doc2vec import Doc2Vec
import nltk
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from transformers import BartForConditionalGeneration, BartTokenizer
import spacy
import os
import datetime
import base64
# import pyautogui
import shutil
import time


st.set_page_config(page_title="Resume Screening Helper")
uploaded_file_path = ""
rest_flag = False


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

st.markdown("""
<style>
.block-container
{
    padding-top: 1rem;
    padding-bottom: 0rem;
    margin-top: 1rem;
}
</style>
""", unsafe_allow_html=True)

if 'session_id' not in st.session_state:
    st.session_state['session_id'] =''



def extract_pdf_data(file_path):
    data = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                data += text
    return data

def set_sidebar_style(side_bg):
   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stSidebar"] > div:first-child {{
          background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
          background-position: center;
          background-size: cover;
      }}
      
      </style>
      """,
      unsafe_allow_html=True,
      )




def set_main_style(side_bg):
   side_bg_ext = 'png'

   st.markdown(
      f"""
      <style>
      [data-testid="stAppViewContainer"] > .main {{
          background-color: #FFD580;
          # background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()});
          # background-position: right center;
          background-size: 100vw 100vh;
          background-size: cover;
          padding: 0;
          # background-attachment: fixed;
          # background-repeat: no-repeat;
          
      }}
      [data-testid="stHeader"] {{
        background: rgba(0,0,0,0);
      }}
      
      
      </style>
      """,
      unsafe_allow_html=True,
      )


def summary_text(resume):
  # sentence-transformers/all-mpnet-base-v2
  model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
  tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")

   # Tokenize the input text
  inputs = tokenizer(resume, max_length=1024, return_tensors="pt", truncation=True)

    # Generate the summary
  summary_ids = model.generate(inputs.input_ids, num_beams=4, max_length=150, early_stopping=True)
  summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
  return summary

def compare(resume_texts, JD_text, flag='HuggingFace-BERT'):
    JD_embeddings = None
    resume_embeddings = []

    if flag == 'HuggingFace-BERT':
        if JD_text is not None:
            JD_embeddings = get_HF_embeddings(JD_text)
        for resume_text in resume_texts:
            
            resume_embeddings.append(get_HF_embeddings(resume_text))

        if JD_embeddings is not None and resume_embeddings is not None:
            cos_scores = cosine(resume_embeddings, JD_embeddings)
            return cos_scores

    # Add logic for other flags like 'Doc2Vec' if necessary
    else:
        # Handle other cases
        pass
     
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


@st.cache_resource
def get_HF_embeddings(sentences):

  # Load model from HuggingFace Hub   bert-large-nli-mean-tokens all-mpnet-base-v2 all-MiniLM-L6-v2 paraphrase-multilingual-MiniLM-L12-v2
  tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
  model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
  # Tokenize sentences
  encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=512)
  # Compute token embeddings
  with torch.no_grad():
      model_output = model(**encoded_input)
  # Perform pooling. In this case, max pooling.
  embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
  
  # print("Sentence embeddings:")
  # print(embeddings)
 
  
  return embeddings

def start_analysis():
  if comp_pressed:
    success_placeholder = st.empty()
    success_placeholder.success("Please enter a job description and upload CV's")

    # Wait for 5 seconds
    time.sleep(5)

              # Remove the success message
    success_placeholder.empty()

def reload_page():
         if reloads:
         
          if 'uploaded_done' not in st.session_state: 
              success_placeholder = st.empty()
              success_placeholder.success("Reset Data Successfully!")
              # Wait for 5 seconds
              time.sleep(5)
              # Remove the success message
              success_placeholder.empty()
          else:           
          #  st.success("Page Reset Successfully")           
            shutil.rmtree(st.session_state.upload)
            st.session_state['uploaded_done'] = False
            st.session_state.upload = ""
            st.markdown("<meta http-equiv='refresh' content='0'>", unsafe_allow_html=True)

@st.cache_data
def get_doc2vec_embeddings(JD, text_resume):
    nltk.download("punkt")
    data = [JD]
    resume_embeddings = []
    
    tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
    #print (tagged_data)

    model = gensim.models.doc2vec.Doc2Vec(vector_size=512, min_count=3, epochs=80)
    model.build_vocab(tagged_data)
    model.train(tagged_data, total_examples=model.corpus_count, epochs=80)
    JD_embeddings = np.transpose(model.docvecs['0'].reshape(-1,1))

    for i in text_resume:
        text = word_tokenize(i.lower())
        embeddings = model.infer_vector(text)
        resume_embeddings.append(np.transpose(embeddings.reshape(-1,1)))
    return (JD_embeddings, resume_embeddings)



def cosine(embeddings1, embeddings2):
  # get the match percentage
  score_list = []
  for i in embeddings1:
      matchPercentage = cosine_similarity(np.array(i), np.array(embeddings2))
      matchPercentage = np.round(matchPercentage, 4)*100 # round to two decimal
      print("Your resume matches about" + str(matchPercentage[0])+ "% of the job description.")
      score_list.append(str(matchPercentage[0][0]))
  return score_list
  

def percentage_to_float(percentage):
    return float(percentage)

def download_pdf(file_contents, file_name):
    # Encode file contents as base64
    encoded_file = base64.b64encode(file_contents).decode()

    # Create data URL
    href = f'<a href="data:application/pdf;base64,{encoded_file}" download="{file_name}">Download {file_name}</a>'
    
    return href

def extract_text_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    return data


# Command-line argument processing
if len(sys.argv) > 1:

    if len(sys.argv) == 3:
        resume_path = sys.argv[1]
        jd_path = sys.argv[2]

        resume_data = extract_pdf_data(resume_path)
        jd_data = extract_text_data(jd_path)

        result = compare([resume_data], jd_data, flag='HuggingFace-BERT')
        
    sys.exit()

# Sidebar
flag = 'HuggingFace-BERT'

with st.sidebar:
    st.markdown("""
    <style>
        [data-testid=stImage] {
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }
    </style>
""", unsafe_allow_html=True)
    set_sidebar_style("sidebar_bg.png")
    st.image("techforce_tf.png")
    desired_count = st.slider("**Number of 'RESUMES' to return**", 1, 50, 2, key="2")
    st.markdown('**Which embedding do you want to use**')
    options = st.selectbox('Which embedding do you want to use',
                           ['HuggingFace-BERT', 'Doc2Vec'],
                           label_visibility="collapsed", disabled=True)
    flag = options
    # st = st.button("Reset")
    # if st:
    #   st.session_state.clear()
    #   try:
    #      pyautogui.hotkey("ctrl","F5")
    #   except KeyError as e:
    #      print(f"KeyError: {e}. Unable to access display.")
       
    
    # st.image("res_tf.png")
    # original_title = '<p style="color:#ff4d04; text-align: center; margin-top: 20px; font-size: 17px;"><b>Techforce Global © 2024 - Version 1.0</b></p>'
    # st.markdown(original_title, unsafe_allow_html=True)
    


# Main content
set_main_style("main_bg.png")
font_css = """
<style>
button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
  font-size: 20px;
  background-color: %s;
  border-radius: 20px;
  padding: 10px;
  
}
</style>
"""

st.write(font_css, unsafe_allow_html=True)
tab1, tab2 = st.tabs(["**🏠 HOME**", "**📝 RESULTS**"])

# Tab Home
with tab1:
    # st.set_page_config(page_title="Resume Screening Helper")
    # title = '<p style="color:#ff4d04; text-align: center; margin-top: 20px; font-size: 25px;"><b>HR - Resume Screening Helper </b></p>'
    # st.markdown(title, unsafe_allow_html=True)
    JD = st.text_area("**Enter the job description:**",height=150)
    uploaded_files = st.file_uploader(
        '**Choose your resume.pdf file:** ', type="pdf", accept_multiple_files=True)
        
    if 'uploaded_done' not in st.session_state:     
      if uploaded_files:                                       
                     for pdf in uploaded_files:
                      output_folder = "uploaded_files_" + datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                      os.makedirs(output_folder, exist_ok=True)
                      uploaded_file_path = output_folder
                      file_path = os.path.join(output_folder, pdf.name)
                      with open(file_path, "wb") as f:
                          f.write(pdf.read())
                      st.session_state['uploaded_done'] = True
                      # st.success("Resume Uploaded Successfully!")
                      upload = ""
                      reset_flag = True
                      st.session_state.upload = uploaded_file_path            
    else:
          if uploaded_files:                                       
            for pdf in uploaded_files:
              file_path = os.path.join(st.session_state.upload, pdf.name)
              with open(file_path, "wb") as f:
                  f.write(pdf.read()) 
                      
                          
    col1, col2 = st.columns([5.5,1])

    with col1:
        comp_pressed = st.button("Start Analysis!")                          
    with col2:
        reloads = st.button("Reset Page")
   
        
    
    # reloads = st.button("Reset Page")
          
    if comp_pressed and uploaded_files:
        
        uploaded_file_paths = [extract_pdf_data(
        file) for file in uploaded_files]                          
        score = compare(uploaded_file_paths, JD, flag)
        if score is not None:
          success_placeholder = st.empty()
          success_placeholder.success("Analysis completed successfully! Wait for few seconds for generating results")
          # Wait for 5 seconds
          time.sleep(10)
          # Remove the success message
          success_placeholder.empty()
          success_placeholder.success("Process completed! Please go to results tab")
          # Wait for 5 seconds
          time.sleep(5)
          # Remove the success message
          success_placeholder.empty()
          # st.success("Analysis done! Please go to results tab.")  
    else:
        start_analysis()
    if reloads:
        reload_page()    
        

# Tab Results
with tab2:    
    st.subheader("**Matched Resumes**")
    my_dict = {}
    my_dict_temp = {}
    matched_dict = {}
    if comp_pressed and uploaded_files:
        
        st.balloons()
        for i in range(len(score)):
           my_dict[uploaded_files[i].name] = score[i] 
           my_dict_temp[uploaded_files[i].name] = uploaded_file_paths[i]          
        sorted_data = dict(sorted(my_dict.items(), key=lambda item: percentage_to_float(item[1]), reverse=True)[:desired_count])
        
        for key, value in sorted_data.items():
             
             with st.expander(str(key),expanded=True):
                # st.write("Score is: ", values)
                st.info(f"**JD Match Score**: {value}")
                # summary = summary_text(my_dict_temp[key])
                # st.info(f"**Summary**: {summary}")                                 
                file_path = os.path.join(st.session_state.upload, key)
                with open(file_path, "rb") as file:
                    file_contents = file.read()
                    st.write(download_pdf(file_contents, key), unsafe_allow_html=True)
                
        
