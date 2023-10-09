from flask import Flask, request, jsonify
from user_input_processor import UserInputProcessor

app = Flask(__name__)

@app.route('/get_top_offers', methods=['POST','GET'])
def get_top_offers():
    try:
        # Get user input from the request
        user_input = request.args.get('user_input')

        # Create an instance of UserInputProcessor
        user_processor = UserInputProcessor('cleaned_data.csv')

        # Process user input and get top offers
        result_df = user_processor.process_and_compare(user_input)

        # Convert result to JSON
        result_json = result_df.to_json(orient='records')

        return jsonify({'success': True, 'result': result_json})

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)