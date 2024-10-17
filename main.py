from flask import Flask, render_template, request, jsonify
from flask_openai import OpenAI
import os
import logging
import numpy as np
from scipy.spatial.distance import cosine

# Basic logging configuration
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)
app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')  # Set your OpenAI API key here

# Initialize the OpenAI extension
openai_extension = OpenAI(app)

# Global variables to store the last message embedding and response
last_message_embedding = None
last_response = None

@app.route('/')
def index():
    return render_template('index.html')

def check_if_same(combined_message):
    global last_message_embedding, last_response

    # Embedding the combined message
    try:
        embedding_response = openai_extension.client.embeddings.create(
            model="text-embedding-ada-002",
            input=combined_message
        )
        current_embedding = np.array(embedding_response['data'][0]['embedding'])

        # Calculate similarity if last_message_embedding exists
        if last_message_embedding is not None:
            similarity = 1 - cosine(last_message_embedding, current_embedding)
            if similarity > 0.9:
                # If similar, return True and the similarity score
                return True, similarity, last_response

        # Update the last message embedding and return False if not similar
        last_message_embedding = current_embedding
        return False, 0, None

    except Exception as e:
        logging.error(f"Error during embedding: {str(e)}")
        return False, 0, None

@app.route('/process', methods=['POST'])
def process_message():
    global last_response
    text = request.json.get('text', '')

    if not text:
        return jsonify({'error': 'No text provided'}), 400

    try:
        # New OpenAI call to assess completeness of the user message
        response = openai_extension.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Respond with a number between 0-100 to show likelihood that the user message is complete and can be responded to."""},
                {"role": "user", "content": "Hi."},
                {"role": "user", "content": "100"},
                {"role": "user", "content": "What is your "},
                {"role": "user", "content": "0"},
                {"role": "user", "content": f"{text}"}
            ],
        )
        
        end_likelihood = int(response['choices'][0]['message']['content'])
        logging.info(f"End likelihood: {end_likelihood}")

        # If the likelihood is 90 or above, consider the message complete and skip to similarity check
        if end_likelihood >= 90:
            combined_message = text
        else:
            # Predict the ending of the user message if likelihood is below 90
            response = openai_extension.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": """You are an AI designed to predict the end of a user message. Generate the expected ending of this user message. Do not include a response to the user message."""},
                    {"role": "user", "content": f"{text}"}
                ],
            )
            restofmessage = response['choices'][0]['message']['content']
            combined_message = f"{text} {restofmessage}"

        # Check if the current message is similar to the last one
        is_similar, similarity, saved_response = check_if_same(combined_message)
        if is_similar:
            # Use the last response if similar
            return jsonify({'response': saved_response, 'restofmessage': combined_message, 'similarity': similarity,'end_likelihood': end_likelihood})

        # Generate a response if not similar or message is considered complete
        response = openai_extension.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": """Generate a response to the user message"""},
                {"role": "user", "content": combined_message}
            ],
        )
        
        output = response['choices'][0]['message']['content']

        # Update the last response
        last_response = output

        return jsonify({'response': output, 'restofmessage': combined_message,'end_likelihood': end_likelihood})

    except Exception as e:
        logging.error(f"Error processing message: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80)
