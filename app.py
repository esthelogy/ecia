import streamlit as st
import requests
import re
import logging
from openai import OpenAI
from pinecone import Pinecone
from functools import lru_cache
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

# Configure logging
#logging.basicConfig(level=logging.DEBUG)

# Load API keys from Streamlit secrets
try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
except KeyError as e:
    st.error(f"API key is missing: {e}. Please configure it in the Streamlit app settings under Secrets.")
    logging.error(f"Missing API key: {e}")
    st.stop()

# Hard-coded index name
index_name = "title-index"

# API base URL
api_base_url = st.secrets["API_BASE_URL"]
prod_api_base_url = st.secrets["PROD_API_BASE_URL"]

# Initialize Pinecone
try:
    logging.info("Initializing Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)

    # List all indexes
    logging.info("Listing Pinecone indexes...")
    all_indexes = pc.list_indexes()
    logging.info(f"Raw Pinecone response: {all_indexes}")

    # Debug: Log the type of all_indexes
    logging.info(f"Type of all_indexes: {type(all_indexes)}")

    # Handle the IndexList object properly
    index_names = []
    if isinstance(all_indexes, list):
        index_names = all_indexes
    elif hasattr(all_indexes, 'indexes'):
        # Access the 'indexes' property directly if it exists
        index_names = [index.name for index in all_indexes.indexes]
    else:
        st.warning(f"Unexpected format of the index list: {all_indexes}")

    logging.info(f"Extracted index names: {index_names}")

    # Check for the specific index
    if index_name not in index_names:
        st.warning(f"Index '{index_name}' not found in Pinecone. Available indexes are: {index_names}")
    else:
        index = pc.Index(index_name)
        logging.info(f"Successfully connected to Pinecone index: {index_name}")
        index_description = index.describe_index_stats()
        logging.info(f"Index dimensions: {index_description['dimension']}")
        logging.info(f"Total vectors: {index_description['total_vector_count']}")

except Exception as e:
    st.error(f"Failed to initialize Pinecone: {str(e)}")
    logging.error(f"Exception encountered: {e}", exc_info=True)
    st.write(f"Error type: {type(e).__name__}")
    st.write(f"Error details: {str(e)}")

# Initialize OpenAI (keep this part as is)
try:
    client = OpenAI(api_key=openai_api_key)
except Exception as e:
    st.error(f"Failed to initialize OpenAI: {str(e)}")
    logging.error(f"OpenAI initialization error: {e}")
    st.stop()

# Helper function to handle API responses
def handle_api_response(response):
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as http_err:
        st.error(f"HTTP error occurred: {response.json().get('message', str(http_err))}")
        logging.error(f"HTTP error occurred: {http_err}")
        logging.info(f"Response content: {response.content}")
        return None
    except Exception as err:
        st.error(f"An error occurred: {err}")
        logging.error(f"Unexpected error: {err}")
        logging.info(f"Response content: {response.content}")
        return None
    logging.info(f"Successful API Response content: {response.content}")
    return response.json()

# Caching for embedding generation to improve scalability
@lru_cache(maxsize=1024)
def get_embedding(text, model="text-embedding-3-small"):
    try:
        text = text.replace("\n", " ")
        return client.embeddings.create(input=[text], model=model).data[0].embedding
    except Exception as e:
        st.error("Failed to generate embedding. Please try again later.")
        logging.error(f"Embedding generation error: {e}")
        return None

# Function to check for similar questions in Pinecone
def check_similarity(embedding, threshold=0.6):
    if 'index' not in globals():
        logging.warning("Pinecone index not available. Similarity check skipped.")
        return None, None
    try:
        query_result = index.query(vector=embedding, top_k=1, include_metadata=True)
        if query_result['matches']:
            score = query_result['matches'][0]['score']
            if score > threshold:
                return query_result['matches'][0]['metadata']['question'], score
    except Exception as e:
        logging.error(f"Similarity check error: {e}")
    return None, None
    
# Create Quiz
def create_quiz(quiz_data):
    try:
        response = requests.post(
            f"{api_base_url}/quiz/create",
            json=quiz_data,
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        result = handle_api_response(response)
        if result and result.get("success"):
            st.success("Quiz created successfully!")
            return result.get("quiz")
        else:
            st.error(f"Failed to create quiz: {result.get('message', 'Unknown error')}")
    except Exception as e:
        st.error("Failed to create quiz. Please try again later.")
        logging.error(f"Error in create_quiz: {e}")
    return None

# Fetch All Quizzes
def get_all_quizzes(page: int = 1, size: int = 50):
    url = f"{api_base_url}/quiz/list"
    params = {"page": page, "size": size}
    headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
    try:
        response = requests.get(url, params=params, headers=headers)
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        if response.status_code == 200:
            data = response.json()
            return data.get("quizzes", {}).get("items", [])
        elif response.status_code == 404:
            logging.error("Quizzes not found")
        elif response.status_code == 401:
            logging.error("Not authenticated")
        else:
            logging.error(f"Unexpected error: {response.status_code}")
    except Exception as e:
        logging.error(f"Error in get_all_quizzes: {e}")
        st.error(f"An error occurred: {e}")
    return []

# Fetch quiz details by ID
def get_quiz_details(quiz_id: str):
    url = f"{api_base_url}/quiz/{quiz_id}"
    headers = {
        "Authorization": f"Bearer {st.session_state.get('auth_token', '')}",
        "Accept": "application/json"
    }
    try:
        response = requests.get(url, headers=headers)
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        if response.status_code == 200:
            return response.json().get("quiz", {})
        else:
            st.error(f"Failed to fetch quiz details: {response.status_code}")
            return {}
    except Exception as e:
        logging.error(f"Error in get_quiz_details: {e}")
        st.error(f"An error occurred: {e}")
        return {}

# Update Quiz
def update_quiz(quiz_id: str, quiz_data: dict):
    try:
        response = requests.put(
            f"{api_base_url}/quiz/{quiz_id}",
            json=quiz_data,
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        if response.status_code == 200:
            st.success("Quiz updated successfully!")
            return response.json()
        else:
            st.error(f"Failed to update quiz: {response.status_code}")
            return None
    except Exception as e:
        logging.error(f"Error in update_quiz: {e}")
        st.error(f"An error occurred: {e}")
        return None

# Delete Quiz
def delete_quiz(quiz_id: str):
    try:
        response = requests.delete(
            f"{api_base_url}/quiz/{quiz_id}",
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        result = handle_api_response(response)
        if result and result.get("success"):
            st.success(f"Quiz {quiz_id} deleted successfully!")
            return True
        else:
            st.error(f"Failed to delete quiz {quiz_id}: {result.get('message', 'Unknown error')}")
    except Exception as e:
        st.error(f"Failed to delete quiz {quiz_id}. Please try again later.")
        logging.error(f"Error in delete_quiz: {e}")
    return False

# Upload Puzzle
def upload_puzzle(file, quiz_id: str):
    try:
        files = {"file": file}
        data = {"quiz_id": quiz_id}
        response = requests.post(
            f"{api_base_url}/admin/quiz/puzzle/upload",
            files=files,
            data=data,
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        result = handle_api_response(response)
        if result and result.get("success"):
            st.success("Puzzle uploaded successfully!")
            return result.get("puzzle")
        else:
            st.error(f"Failed to upload puzzle: {result.get('message', 'Unknown error')}")
    except Exception as e:
        st.error("Failed to upload puzzle. Please try again later.")
        logging.error(f"Error in upload_puzzle: {e}")
    return None

# List Estheticians
def list_estheticians(page: int = 1, limit: int = 10) -> List[Dict[str, Any]]:
    try:
        response = requests.get(
            f"{api_base_url}/admin/esthetician/approval_list",
            params={"page": page, "limit": limit},
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        print(f"{api_base_url}/admin/esthetician/approval_list")
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        result = handle_api_response(response)
        if result and result.get("success"):
            return result.get("estheticians", [])
        else:
            st.error("Failed to fetch estheticians. Please try again later.")
            return []
    except Exception as e:
        st.error("An error occurred while fetching estheticians.")
        logging.error(f"Error in list_estheticians: {e}")
        return []

# Approve Esthetician
def approve_esthetician(esthetician_id: str, is_approved: bool = True, reason_for_rejection: str = "N/A") -> str:
    try:
        st.write(f"Reason for rejection: {reason_for_rejection}")
        is_approved_str = "true" if is_approved else "false"
        request_body = {
            "is_approved": is_approved_str,
            "reason_for_rejection": reason_for_rejection if not is_approved else "N/A"
        }
        logging.info(f"Request body for approving esthetician: {request_body}")
        response = requests.put(
            f"{api_base_url}/admin/approve_esthetician/{esthetician_id}",
            json=request_body,
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        logging.info(f"Response status code: {response.status_code}")
        logging.info(f"Response content: {response.content}")
        result = handle_api_response(response)
        if result and result.get("success"):
            status = "approved" if is_approved else "rejected"
            st.success(f"Esthetician {esthetician_id} {status} successfully.")
            return "true"
        else:
            st.error(f"Failed to update esthetician {esthetician_id}.")
            return "false"
    except Exception as e:
        st.error(f"An error occurred while updating esthetician {esthetician_id}.")
        logging.error(f"Error in approve_esthetician: {e}")
        return "false"

# Show Admin Page
def show_admin_page():
    st.title("Admin Page")
    st.subheader("Menu")
    
    if st.button("Esthetician Management"):
        st.session_state["environment"] = "dev"
        st.session_state["page"] = "esthetician_management"
    
    if st.button("Esthetician Management (Prod)"):
        st.session_state["environment"] = "prod"
        st.session_state["page"] = "esthetician_management"
    
    if st.button("Quiz Management"):
        st.session_state["page"] = "quiz_management"

    if st.button("Logout"):
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.success("You have been logged out.")

def analyze_skin_condition(text_input: str, image_file=None):
    url = f"{api_base_url}/ai/analyze_skin"
    
    if image_file is not None:
        files = {
            'image': image_file
        }
        data = {
            'text_input': text_input
        }
        response = requests.post(url, files=files, data=data)
    else:
        payload = {
            "text_input": text_input
        }
        response = requests.post(url, json=payload)

    print(response.text)
    if response.status_code == 200:
        return response.json()
    else:
        return None

def get_quiz_recommendations(skin_type: str, condition: str):
    url = f"{api_base_url}/ai/diary_quiz_recommendation"
    payload = {
        "skin_type": skin_type,
        "condition": condition
    }
    headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}

    response = requests.post(url, json=payload, headers=headers)

    print(response.text)
    if response.status_code == 200:
        return response.json().get('quiz_list', [])
    else:
        print(f"Failed to fetch quiz recommendations: {response.status_code} - {response.text}")
        return None
    
def start_quiz(quiz_id: str):
    print("Starting Quiz")
    try:
        url = f"{api_base_url}/quiz/start_quiz"
        params = {"quiz_id": quiz_id}  
        headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        response = requests.get(url, headers=headers, params=params)

        if response.status_code == 200:
            quiz_details = response.json().get('question_details', {})
            logging.info(f"Quiz started successfully: {quiz_details}")
            return quiz_details
        else:
            st.error(f"Failed to start quiz: {response.status_code} - {response.text}")
            logging.error(f"Failed to start quiz: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(str(e))

def fetch_next_question(quiz_id: str, question_id: str, response: str, response_time: int):
    url = f"{api_base_url}/quiz/fetch_next"
    payload = {
        "quiz_id": quiz_id,
        "question_id": question_id,
        "response": response,
        "response_time": response_time
    }
    headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json().get('question_details', {})
    else:
        st.error(f"Failed to fetch next question: {response.status_code} - {response.text}")
        return None

def submit_quiz(quiz_id: str):
    url = f"{api_base_url}/quiz/submit"
    payload = {"quiz_id": quiz_id}
    headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        return response.json()
    else:
        st.error(f"Failed to submit quiz: {response.status_code} - {response.text}")
        return None

def show_recommendations_page():
    st.title("Quiz Recommendations")

    if 'quiz_recommendations' in st.session_state and st.session_state['quiz_recommendations']:
        st.subheader("Here are your quiz recommendations based on your skin analysis:")
        for recommendation in st.session_state['quiz_recommendations']:
            st.write(f"- {recommendation}")
    else:
        st.info("No recommendations available. Please complete the skin analysis first.")

    if st.button("Back to Main"):
        st.session_state["page"] = "main"

def get_quiz_details(quiz_id: str):
    url = f"{api_base_url}/quiz/{quiz_id}"
    headers = {"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
    response = requests.get(url, headers=headers)

    if response.status_code == 200 and response.json().get('success'):
        return response.json().get('quiz', {})
    else:
        st.error(f"Failed to fetch quiz details: {response.status_code} - {response.text}")
        return None
    
def show_quiz_details_page(quiz_id: str):
    st.title("Quiz Details")

    # Fetch the quiz details using the provided API
    quiz = get_quiz_details(quiz_id)

    if quiz:
        st.subheader(quiz.get("title", "Quiz Title"))
        st.write(f"Section: {quiz.get('section', 'N/A')}")

        questions = quiz.get("questions", [])
        if questions:
            for idx, question in enumerate(questions):
                st.markdown(f"**Question {idx + 1}:** {question.get('question', 'N/A')}")
                options = question.get("options", [])
                for option in options:
                    st.radio(f"Question {idx + 1}", options, key=f"option_{question['question_id']}_{option}")
                st.markdown("---")
        else:
            st.write("No questions available in this quiz.")
    else:
        st.error("Failed to retrieve quiz details.")

def show_chatroom_page():
    st.title("My Skin Diary")

    st.subheader("How is your skin today?")
    diary_entry = st.text_area("Share your observation, thought, and feeling about your skin today")
    media_file = st.file_uploader("Upload a picture to help us understand better", type=["jpg", "jpeg", "png"])

    if st.button("Analyze Skin"):
        if diary_entry or media_file:
            response = analyze_skin_condition(diary_entry, media_file)

            if response and response.get("success"):
                detail = response.get("detail", {})
                skin_type = detail.get('skin_type')
                condition = detail.get('condition')

                # Store skin analysis results and recommendations in session state
                st.session_state['skin_type'] = skin_type
                st.session_state['condition'] = condition
                st.session_state['skin_recommendations'] = detail.get("recommendations", [])

                st.subheader("Skin Analysis Results")
                st.write(f"**Skin Type:** {skin_type}")
                st.write(f"**Condition:** {condition}")

                # Call the quiz recommendations API with the auth token
                quiz_list = get_quiz_recommendations(skin_type, condition)
                st.session_state['quiz_list'] = quiz_list

                st.subheader("Quiz Recommendations")
                if quiz_list:
                    for i, quiz in enumerate(quiz_list):
                        st.markdown("---")
                        st.write(f"ID: {quiz['quiz_id']}")
                        st.write(f"**{quiz['title']}** ({quiz['section']})")
                        st.write(f"Status: {quiz['status'].replace('_', ' ').capitalize()}")

                        if st.button(f"Start Quiz - {quiz['title']}", key=f"start_quiz_{i}"):
                            st.session_state["quiz_id"] = quiz['quiz_id']
                            st.session_state["page"] = "quiz"
                            st.session_state["current_question"] = start_quiz(quiz['quiz_id'])
                            return

                else:
                    st.write("No quiz recommendations available.")

                st.success("Analysis complete!")
            else:
                st.error("Failed to analyze the skin condition. Please try again.")
        else:
            st.error("Please provide either a text description or an image for analysis.")

    # Navigation Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Find More Insights"):
            st.session_state["page"] = "insights"
    with col2:
        if st.button("Logout"):
            st.session_state.clear()
            st.session_state["page"] = "login"

            if st.button(f"Start Quiz - {quiz['title']}", key=f"start_quiz_{i}"):
                st.session_state["quiz_id"] = quiz['quiz_id']
                st.session_state["page"] = "quiz"
                st.write(f"Quiz '{quiz['title']}' selected.")

def show_quiz_page():
    st.title("Quiz")

    if "current_question" not in st.session_state:
        st.error("No quiz started. Please select a quiz to start.")
        return

    # Get the current question details
    question_details = st.session_state["current_question"]
    question = question_details["question"]
    quiz_id = question_details["quiz_id"]

    # Display the current question and options
    st.subheader(f"Question {question_details['current_question_index'] + 1}")
    st.write(question["question"])

    selected_option = st.radio("Select an option", question["options"])
    response_time = st.slider("Response Time (seconds)", min_value=1, max_value=60, value=10)

    if st.button("Submit Answer"):
        # Handle submitting the current question's answer
        next_question_details = fetch_next_question(
            quiz_id=quiz_id,
            question_id=question["question_id"],
            response=selected_option,
            response_time=response_time
        )

        if next_question_details:
            st.session_state["current_question"] = next_question_details

            # Check if this is the last question
            if next_question_details["current_question_index"] + 1 >= next_question_details["total_question_count"]:
                st.success("All questions answered. Submitting the quiz...")
                result = submit_quiz(quiz_id)
                if result and result.get("success"):
                    st.session_state["page"] = "insights"
                else:
                    st.error("Failed to submit the quiz. Please try again.")
            else:
                st.query_params.update(rerun=True)

        else:
            st.error("Failed to fetch the next question. Please try again.")

def show_quiz_results_page():
    st.title("Quiz Submitted")

    st.write("Thank you for completing the quiz! Your responses have been recorded.")

    if st.button("Back to Main"):
        st.session_state["page"] = "main"

def show_insights_page():
    st.title("Find More Insights")

    if 'skin_recommendations' in st.session_state:
        st.subheader("Your Skin Recommendations:")
        for recommendation in st.session_state['skin_recommendations']:
            st.write(f"- {recommendation}")
    else:
        st.info("No skin recommendations available. Please perform skin analysis first.")

    if st.button("Logout"):
        st.session_state.clear()
        st.session_state["page"] = "login"

# Show Quiz Management
def show_quiz_management():
    st.title("Quiz Management")
    page = st.number_input("Page", min_value=1, value=1)
    size = st.number_input("Quizzes per page", min_value=1, value=10)

    quizzes = get_all_quizzes(page, size)

    if quizzes:
        for quiz_idx, quiz in enumerate(quizzes):
            quiz_id = quiz.get("_id")
            st.write(f"ID: {quiz_id}")
            if quiz_id:
                quiz_details = get_quiz_details(quiz_id)
                title = quiz_details.get("title", "N/A")
                section = quiz_details.get("section", "N/A")
                questions = quiz_details.get("questions", [])

                st.markdown(f"### {title}")
                st.markdown(f"**Section**: {section}")

                edited_questions = []
                for q_idx, question in enumerate(questions):
                    question_text = st.text_area(f"Question: {question.get('question', 'N/A')}", value=question.get('question', 'N/A'), key=f"{title}_question_{q_idx}")
                    options = question.get("options", [])
                    edited_options = [st.text_area(f"Option {i+1}", value=option, key=f"{title}_question_{q_idx}_option_{i}") for i, option in enumerate(options)]

                    # Delete question button
                    if st.button(f"Delete Question {q_idx + 1}", key=f"delete_{title}_{q_idx}"):
                        st.session_state['delete_confirm'] = {
                            "quiz_id": quiz_id,
                            "question_idx": q_idx
                        }

                    # Check if a delete confirmation is required
                    if 'delete_confirm' in st.session_state and st.session_state['delete_confirm']['quiz_id'] == quiz_id and st.session_state['delete_confirm']['question_idx'] == q_idx:
                        st.warning(f"Are you sure you want to delete this question: '{question_text}'?")
                        if st.button("Confirm Delete", key=f"confirm_delete_{title}_{q_idx}"):
                            st.session_state.pop('delete_confirm')
                            # Remove the question from the list
                            questions.pop(q_idx)
                            updated_quiz = {
                                "title": title,
                                "section": section,
                                "questions": questions
                            }
                            update_quiz(quiz_id, updated_quiz)
                            st.query_params.update(rerun=True)

                        if st.button("Cancel", key=f"cancel_delete_{title}_{q_idx}"):
                            st.session_state.pop('delete_confirm')

                    # Add the question to the list only if it wasn't deleted
                    if not (st.session_state.get('delete_confirm', {}).get('question_idx') == q_idx):
                        edited_questions.append({
                            "question_id": question.get("question_id"),
                            "question": question_text,
                            "options": edited_options
                        })

                if st.button("Save", key=f"save_{title}_{quiz_idx}"):
                    updated_quiz = {
                        "title": title,
                        "section": section,
                        "questions": edited_questions
                    }
                    update_quiz(quiz_id, updated_quiz)

                if st.button("Discard", key=f"discard_{title}_{quiz_idx}"):
                    st.query_params.update(rerun=True)

    if st.button("Back to Admin Page", key="back_to_admin"):
        st.session_state["page"] = "admin"


# Create Quiz Page
def create_quiz_page():
    st.title("Create a New Quiz")
    st.write("You can add sections, subsections, and questions. Each question will be checked for similarity before being added.")

    if "new_quiz" not in st.session_state:
        st.session_state["new_quiz"] = {
            "quiz_title": "",
            "sections": []
        }

    st.session_state["new_quiz"]["quiz_title"] = st.text_input("Enter Quiz Title", value=st.session_state["new_quiz"]["quiz_title"], help="The title of the quiz.")
    new_section_name = st.text_input("Enter Section Name", help="Add a new section to your quiz.")
    new_subsection_name = st.text_input("Enter Subsection Name", help="Add a new subsection under the section.")

    if st.button("Add Section"):
        if new_section_name:
            st.session_state["new_quiz"]["sections"].append({
                "section_name": new_section_name,
                "subsections": []
            })
            st.success(f"Section '{new_section_name}' added.")
        else:
            st.error("Please enter a section name.")

    if st.button("Add Subsection"):
        if new_subsection_name and st.session_state["new_quiz"]["sections"]:
            st.session_state["new_quiz"]["sections"][-1]["subsections"].append({
                "subsection_name": new_subsection_name,
                "questions": []
            })
            st.success(f"Subsection '{new_subsection_name}' added to Section '{st.session_state['new_quiz']['sections'][-1]['section_name']}'.")
        else:
            st.error("Please enter a subsection name or add a section first.")

    for section_idx, section in enumerate(st.session_state["new_quiz"]["sections"]):
        st.subheader(f"Section {section_idx + 1}: {section['section_name']}")
        for subsection_idx, subsection in enumerate(section["subsections"]):
            st.text(f"Subsection {subsection_idx + 1}: {subsection['subsection_name']}")

            question_text = st.text_input(f"Enter a question for {subsection['subsection_name']}:", key=f"question_{section_idx}_{subsection_idx}")
            if st.button(f"Add Question to {subsection['subsection_name']}", key=f"add_question_{section_idx}_{subsection_idx}"):
                if question_text:
                    embedding = get_embedding(question_text)
                    similar_question, score = check_similarity(embedding)
                    if similar_question:
                        st.warning(f"This question is similar to an existing question: '{similar_question}' with a similarity score of {(score*100):.2f}. Consider revising it.")
                    else:
                        subsection["questions"].append(question_text)
                        st.success(f"Question added to {subsection['subsection_name']}.")
                else:
                    st.error("Please enter a question.")

    if st.button("Save Quiz"):
        if create_quiz(st.session_state["new_quiz"]):
            st.success("Quiz saved successfully!")
            st.session_state.pop("new_quiz")
        else:
            st.error("Failed to save the quiz. Please try again later.")

    if st.button("Back"):
        st.session_state["page"] = "quiz_management"

# Edit Quiz Page
def edit_quiz_page():
    if "editing_quiz" not in st.session_state:
        st.error("No quiz selected for editing.")
        st.session_state["page"] = "quiz_management"
        return

    quiz = st.session_state["editing_quiz"]
    st.title(f"Edit Quiz: {quiz['title']}")

    # Edit quiz title
    new_title = st.text_input("Quiz Title", value=quiz['title'])

    # Check if 'section' key exists in the quiz data and is a string
    if 'section' not in quiz or not isinstance(quiz['section'], str):
        st.error("Quiz data is missing section or section is not a string.")
        return

    # Extract sections from the Pinecone data
    sections = [{"name": quiz['section'], "subsections": [{"name": "Default Subsection", "questions": quiz.get('questions', [])}]}]

    # Edit sections and subsections
    updated_sections = []
    for section in sections:
        st.subheader(f"Section: {section['name']}")
        new_section_name = st.text_input("Section Name", value=section['name'], key=f"section_{section['name']}")

        updated_subsections = []
        for subsection in section['subsections']:
            st.text(f"Subsection: {subsection['name']}")
            new_subsection_name = st.text_input("Subsection Name", value=subsection['name'], key=f"subsection_{subsection['name']}")

            updated_questions = []
            for idx, question in enumerate(subsection['questions']):
                new_question_text = st.text_area("Question", value=question, key=f"question_{idx}")
                if new_question_text != question:
                    updated_questions.append({"id": idx, "text": new_question_text})

            if new_subsection_name != subsection['name'] or updated_questions:
                updated_subsections.append({
                    "name": new_subsection_name,
                    "questions": updated_questions
                })

        if new_section_name != section['name'] or updated_subsections:
            updated_sections.append({
                "name": new_section_name,
                "subsections": updated_subsections
            })

    if st.button("Save Changes"):
        updated_quiz = {
            "id": quiz['_id'],
            "title": new_title,
            "section": updated_sections
        }
        if update_quiz(quiz['_id'], updated_quiz):
            st.success("Quiz updated successfully!")
            st.session_state.pop("editing_quiz")
            st.session_state["page"] = "quiz_management"
        else:
            st.error("Failed to update quiz. Please try again.")

    if st.button("Cancel"):
        st.session_state.pop("editing_quiz")
        st.session_state["page"] = "quiz_management"

def list_estheticians_env(
    page: int = 1,
    limit: int = 10,
    environment: str = "dev"
) -> List[Dict[str, Any]]:
    """
    Fetch estheticians from either the dev/test or production environment for approval.
    """

    if environment == "prod":
        base_url = prod_api_base_url
        token = st.session_state.get("auth_token_prod", "")
    else:
        base_url = api_base_url
        token = st.session_state.get("auth_token", "")
    endpoint = f"{base_url}/admin/esthetician/approval_list"
    try:
        response = requests.get(
            endpoint,
            params={"page": page, "limit": limit},
            headers={"Authorization": f"Bearer {token}"}
        )
        logging.info(f"[{environment.upper()}] Response status code: {response.status_code}")
        logging.info(f"[{environment.upper()}] Response content: {response.content}")

        result = handle_api_response(response)
        if result and result.get("success"):
            return result.get("estheticians", [])
        else:
            st.error(f"[{environment.upper()}] Failed to fetch estheticians. Please try again later.")
            return []
    except Exception as e:
        st.error(f"[{environment.upper()}] An error occurred while fetching estheticians.")
        logging.error(f"[{environment.upper()}] Error in list_estheticians_env: {e}")
        return []

def approve_esthetician_env(
    esthetician_id: str,
    is_approved: bool = True,
    reason_for_rejection: str = "N/A",
    environment: str = "dev"
) -> str:
    """
    Approve or reject an esthetician in either the dev/test or production environment.
    """
    base_url = get_base_url(environment)
    endpoint = f"{base_url}/admin/approve_esthetician/{esthetician_id}"

    try:
        is_approved_str = "true" if is_approved else "false"
        request_body = {
            "is_approved": is_approved_str,
            "reason_for_rejection": reason_for_rejection if not is_approved else "N/A"
        }

        logging.info(f"[{environment.upper()}] Request body for approving esthetician: {request_body}")
        response = requests.put(
            endpoint,
            json=request_body,
            headers={"Authorization": f"Bearer {st.session_state.get('auth_token', '')}"}
        )
        logging.info(f"[{environment.upper()}] Response status code: {response.status_code}")
        logging.info(f"[{environment.upper()}] Response content: {response.content}")

        result = handle_api_response(response)
        if result and result.get("success"):
            status = "approved" if is_approved else "rejected"
            st.success(f"[{environment.upper()}] Esthetician {esthetician_id} {status} successfully.")
            return "true"
        else:
            st.error(f"[{environment.upper()}] Failed to update esthetician {esthetician_id}.")
            return "false"
    except Exception as e:
        st.error(f"[{environment.upper()}] An error occurred while updating esthetician {esthetician_id}.")
        logging.error(f"[{environment.upper()}] Error in approve_esthetician_env: {e}")
        return "false"

# Show Esthetician Management
def show_esthetician_management():
    """
    A unified page that displays and manages estheticians in either dev or prod,
    depending on st.session_state['environment'].
    """
    # Default to dev if none is set
    environment = st.session_state.get("environment", "dev")
    st.title(f"Esthetician Management - {environment.upper()}")

    page = st.number_input("Page", min_value=1, value=1)
    limit = st.number_input("Estheticians per page", min_value=1, max_value=100, value=10)

    # Fetch estheticians based on environment
    estheticians = list_estheticians_env(page, limit, environment=environment)

    if estheticians:
        for esthetician in estheticians:
            st.write(f"ID: {esthetician['_id']}")
            st.write(f"Name: {esthetician['full_name']}, License No: {esthetician['license_no']}")
            st.write(f"Email: {esthetician['email']}, Status: {esthetician['esthetician_status']}")
            
            # If there's a license_file, display a snippet:
            if esthetician.get("license_file") and "data" in esthetician["license_file"]:
                st.write(f"License File: {esthetician['license_file']['data']}")

            # Show reason for rejection if available
            if esthetician.get("reason_for_rejection", "N/A") != "N/A":
                st.write(f"Reason for Rejection: {esthetician['reason_for_rejection']}")

            # If not approved, show approve/reject buttons
            if esthetician['esthetician_status'] != 'approved':
                col1, col2 = st.columns(2)

                with col1:
                    if st.button(
                        f"Approve {esthetician['full_name']} ({environment.upper()})",
                        key=f"approve_{environment}_{esthetician['_id']}"
                    ):
                        # Approve
                        if approve_esthetician_env(esthetician['_id'], True, "N/A", environment=environment) == "true":
                            st.experimental_rerun()

                with col2:
                    reason_for_rejection = st.text_input(
                        f"Reason for rejecting {esthetician['full_name']} ({environment.upper()})",
                        key=f"reason_{environment}_{esthetician['_id']}"
                    )
                    if st.button(
                        f"Reject {esthetician['full_name']} ({environment.upper()})",
                        key=f"reject_{environment}_{esthetician['_id']}"
                    ):
                        # Reject
                        if approve_esthetician_env(
                            esthetician['_id'],
                            is_approved=False,
                            reason_for_rejection=reason_for_rejection,
                            environment=environment
                        ) == "true":
                            st.experimental_rerun()

            st.markdown("---")
    else:
        st.info(f"[{environment.upper()}] No estheticians found or failed to fetch the list.")

    if st.button("Back to Admin Page"):
        st.session_state["page"] = "admin"

# Show Login Page
def show_login_page():
    st.title("Esthelogy Admin")

    username = st.text_input("Email")
    password = st.text_input("Password", type="password")
    api_url = f"{api_base_url}/user/login"

    if st.button("Login"):
        logging.info(f"Attempting to login with email: {username}")
        auth_response = authenticate(username, password, api_url)
        logging.info(f"Authentication response: {auth_response}")
        if auth_response:
            if auth_response.get("success"):
                st.success("Login successful!")
                st.session_state["auth_token"] = auth_response.get("access_token", "")
                st.session_state["user_id"] = auth_response.get("user_id", "")
                user_role = auth_response.get("role", "")
                
                if user_role == "admin":
                    st.session_state["page"] = "admin"
                else:
                    st.session_state["page"] = "chatroom"
            else:
                st.error(f"Login failed: {auth_response.get('message')}")
        else:
            st.error("Login failed. Please check your credentials or API URL.")

def show_login_page():
    st.title("Esthelogy Admin")

    username = st.text_input("Email")
    password = st.text_input("Password", type="password")

    # Existing Dev API URL
    dev_api_url = f"{api_base_url}/user/login"
    # New: Production API URL
    prod_api_url = f"{prod_api_base_url}/user/login"

    if st.button("Login"):
        logging.info(f"Attempting to login (Dev) with email: {username}")
        dev_auth_response = authenticate(username, password, dev_api_url)
        logging.info(f"Dev Authentication response: {dev_auth_response}")

        # Check if Dev login succeeded
        if dev_auth_response and dev_auth_response.get("success"):
            st.success("Dev login successful!")
            st.session_state["auth_token"] = dev_auth_response.get("access_token", "")
            st.session_state["user_id"] = dev_auth_response.get("user_id", "")
            user_role = dev_auth_response.get("role", "")
        else:
            st.error("Dev login failed. Please check your credentials or Dev API.")

        # Now log into Prod
        logging.info(f"Attempting to login (Prod) with email: {username}")
        prod_auth_response = authenticate(username, password, prod_api_url)
        logging.info(f"Prod Authentication response: {prod_auth_response}")

        # Check if Prod login succeeded
        if prod_auth_response and prod_auth_response.get("success"):
            st.success("Prod login successful!")
            # Store token in a separate key so you can differentiate
            st.session_state["auth_token_prod"] = prod_auth_response.get("access_token", "")
            st.session_state["user_id_prod"] = prod_auth_response.get("user_id", "")
        else:
            st.warning("Prod login failed. You may not be able to manage Prod data.")

        if user_role == "admin":
            st.session_state["page"] = "admin"
        else:
            st.session_state["page"] = "chatroom"

# Authenticate function
def authenticate(username, password, api_url):
    payload = {"email": username, "password": password}
    try:
        st.write(f"Sending authentication request to: {api_url}")
        response = requests.post(api_url, json=payload)
        st.write(f"Authentication response status code: {response.status_code}")
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Authentication error: {str(e)}")
        return None

# Define Navigation Menu
def show_navigation_menu():
    st.sidebar.title("Navigation")
    if st.sidebar.button("Admin Page", key="nav_admin_page"):
        st.session_state["page"] = "admin"
    if st.sidebar.button("Esthetician Management", key="nav_esthetician_management"):
        st.session_state["environment"] = "dev"
        st.session_state["page"] = "esthetician_management"
    if st.sidebar.button("Esthetician Management (Prod)", key="nav_prod_esthetician_management"):
        st.session_state["environment"] = "prod"
        st.session_state["page"] = "esthetician_management"
    if st.sidebar.button("Quiz Management", key="nav_quiz_management"):
        st.session_state["page"] = "quiz_management"
    if st.sidebar.button("Create Quiz", key="nav_create_quiz"):
        st.session_state["page"] = "create_quiz"
    if st.sidebar.button("Chatroom", key="nav_chatroom"):
        st.session_state["page"] = "chatroom"
    if st.sidebar.button("Logout", key="nav_logout"):
        st.session_state.clear()
        st.session_state["page"] = "login"
        st.success("You have been logged out.")

# Main function to control the app flow
def main():
    #for debugging
    # Sample log entry
    log_entry = "2024-08-25 04:11:52,951 INFO Total quizzes: 0"

    # Define the regular expression pattern
    pattern = r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}) (\w+) (.+)'

    # Use re.findall to extract information
    matches = re.findall(pattern, log_entry)

    # Process the extracted information
    for match in matches:
        timestamp, log_level, message = match
        print(f"Timestamp: {timestamp}")
        print(f"Log Level: {log_level}")
        print(f"Message: {message}")

    # Original code
    if "page" not in st.session_state:
        st.session_state["page"] = "login"

    show_navigation_menu()  # Display the navigation menu

    if st.session_state["page"] == "login":
        show_login_page()
    elif st.session_state["page"] == "admin":
        show_admin_page()
    elif st.session_state["page"] == "quiz_management":
        show_quiz_management()
    elif st.session_state["page"] == "create_quiz":
        create_quiz_page()
    elif st.session_state["page"] == "edit_quiz":
        edit_quiz_page()
    elif st.session_state["page"] == "esthetician_management":
        show_esthetician_management()
    elif st.session_state["page"] == "chatroom":
        show_chatroom_page()
    elif st.session_state["page"] == "quiz":
        show_quiz_page()
    elif st.session_state["page"] == "view_quiz":
        show_quiz_details_page(st.session_state["quiz_id"])
    elif st.session_state["page"] == "insights":
        show_insights_page()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
