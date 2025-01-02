%pip install transformers datasets optuna torch accelerate>=0.26.0

%run ./ts_analysis

# Catbot Code
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset
import torch
import re

# 1. Preprocess the text data
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)  # Remove punctuation
    return text

# 2. Define dataset and intents
data = [
    # Intent: set_data
    {"text": preprocess_text("Set data for stock XYZ"), "intent": "set_data"},
    {"text": preprocess_text("Input data for stock ABC"), "intent": "set_data"},
    {"text": preprocess_text("Configure the data for stock DEF"), "intent": "set_data"},
    {"text": preprocess_text("Set up stock data for GHI"), "intent": "set_data"},
    {"text": preprocess_text("Load data for stock XYZ"), "intent": "set_data"},
    {"text": preprocess_text("Set data XYZ"), "intent": "set_data"},
    {"text": preprocess_text("Choose stock ABC"), "intent": "set_data"},

    # Intent: save
    {"text": preprocess_text("Save stock data for ABC"), "intent": "save"},
    {"text": preprocess_text("Store the stock data for DEF"), "intent": "save"},
    {"text": preprocess_text("Persist the data for stock XYZ"), "intent": "save"},
    {"text": preprocess_text("Save the current data for stock GHI"), "intent": "save"},
    {"text": preprocess_text("Record stock data for ABC"), "intent": "save"},
    {"text": preprocess_text("Save ABC"), "intent": "save"},
    {"text": preprocess_text("ADD stock ABC"), "intent": "save"},

    # Intent: analysis_predict
    {"text": preprocess_text("What is the prediction for stock XYZ?"), "intent": "analysis_predict"},
    {"text": preprocess_text("Give me the forecast for stock ABC"), "intent": "analysis_predict"},
    {"text": preprocess_text("Predict the outcome for stock DEF"), "intent": "analysis_predict"},
    {"text": preprocess_text("What are the analytics for stock XYZ?"), "intent": "analysis_predict"},
    {"text": preprocess_text("Show the prediction results for stock GHI"), "intent": "analysis_predict"},
    {"text": preprocess_text("Prediction results for stock XYZ"), "intent": "analysis_predict"},
    {"text": preprocess_text("Predict results for ABC"), "intent": "analysis_predict"},
    {"text": preprocess_text("Predict GHI"), "intent": "analysis_predict"},

    # Intent: greeting
    {"text": preprocess_text("Hello there!"), "intent": "greeting"},
    {"text": preprocess_text("Hi! How are you?"), "intent": "greeting"},
    {"text": preprocess_text("Hey! Nice to meet you."), "intent": "greeting"},
    {"text": preprocess_text("Good morning!"), "intent": "greeting"},
    {"text": preprocess_text("Good evening! How’s it going?"), "intent": "greeting"},
    {"text": preprocess_text("Good day"), "intent": "greeting"},
    {"text": preprocess_text("Hi"), "intent": "greeting"},
    {"text": preprocess_text("Hi, could you help"), "intent": "greeting"},

    # Intent: thank_you
    {"text": preprocess_text("Thank you so much!"), "intent": "thank_you"},
    {"text": preprocess_text("Thanks a lot for your help!"), "intent": "thank_you"},
    {"text": preprocess_text("I really appreciate it!"), "intent": "thank_you"},
    {"text": preprocess_text("Thanks, you're the best!"), "intent": "thank_you"},
    {"text": preprocess_text("Much obliged!"), "intent": "thank_you"},
    {"text": preprocess_text("Thank you!"), "intent": "thank_you"},
    {"text": preprocess_text("Thanks!"), "intent": "thank_you"},

    # Intent: good_bye
    {"text": preprocess_text("Goodbye, see you later!"), "intent": "good_bye"},
    {"text": preprocess_text("Bye for now!"), "intent": "good_bye"},
    {"text": preprocess_text("See you next time!"), "intent": "good_bye"},
    {"text": preprocess_text("Farewell, take care!"), "intent": "good_bye"},
    {"text": preprocess_text("Catch you later!"), "intent": "good_bye"},
    {"text": preprocess_text("Bye!"), "intent": "good_bye"},
    {"text": preprocess_text("Bye, thank you!"), "intent": "good_bye"},
]

intents = ["set_data", "save", "analysis_predict", "greeting", "thank_you", "good_bye"]
label_map = {intent: idx for idx, intent in enumerate(intents)}

# 3. Add labels to the dataset
for item in data:
    item['label'] = label_map[item['intent']]

dataset = Dataset.from_list(data)

# 4. Load tokenizer and model
model_name = "bert-base-uncased"  # Replace with a more powerful model if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(intents))

# Add stock-specific tokens
tokenizer.add_tokens(["stock", "XYZ", "ABC", "DEF", "GHI"])
model.resize_token_embeddings(len(tokenizer))

# Set id2label and label2id mappings
model.config.id2label = {i: intent for i, intent in enumerate(intents)}
model.config.label2id = {intent: i for i, intent in enumerate(intents)}

# 5. Preprocess data for training
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding=True)

tokenized_data = dataset.map(preprocess_function, batched=True)

# 6. Training arguments and Trainer
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    eval_dataset=tokenized_data,
)

# Train the model
trainer.train()

# Save model and tokenizer
model.save_pretrained("./my_stock_intent_model")
tokenizer.save_pretrained("./my_stock_intent_model")

# 7. Prediction with confidence
def predict_intent_with_confidence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, predicted_class_idx].item()
    return model.config.id2label[predicted_class_idx], confidence

# text = "What is the prediction for stock XYZ?"
# intent, confidence = predict_intent_with_confidence(text)
# print(f"Predicted intent: {intent}, Confidence: {confidence:.2f}")


# Chatbot Instance
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# 1. Load Pretrained Model and Tokenizer
model_path = "./my_stock_intent_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# 2. Define the Prediction Function
def predict_intent_with_confidence(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1)
    predicted_class_idx = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, predicted_class_idx].item()
    return model.config.id2label[predicted_class_idx], confidence

# 3. Chatbot Functionality
def chatbot():
    print("Welcome to the Stock Intent Chatbot! (Type 'exit' to stop)\n")
    
    stock_forecast = StockForecasting(api_key="YOUR_ALPHA_VANTAGE_API_KEY")  # Replace with your API key
    stock_data = None
    symbol = None

    while True:
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Bot: Goodbye! Have a great day!")
            break
        
        if user_input == "":
            print("Bot: Please say something!")
            continue
        
        # Predict the intent
        intent, confidence = predict_intent_with_confidence(user_input)
        
        # Respond based on intent
        if intent == "set_data":
            symbol = input("Bot: Please enter the stock symbol (e.g., AAPL): ").strip().upper()
            if symbol:
                print(f"Bot: Fetching data for {symbol}...")
                stock_data = stock_forecast.fetch_stock_data(symbol)
                if stock_data is not None:
                    print("Bot: Stock data fetched successfully!")
                else:
                    print("Bot: Unable to fetch stock data. Please check the symbol and try again.")
        
        elif intent == "save":
            if stock_data is not None:
                stock_forecast.save_to_delta(stock_data)
                stock_forecast.plot_stock_data(symbol)
                print(f"Bot: Stock data for {symbol} has been saved and plotted successfully.")
            else:
                print("Bot: No stock data to save. Please set the data first.")
        
        elif intent == "analysis_predict":
            if symbol:
                print(f"Bot: Analyzing and predicting stock data for {symbol}...")
                best_model, df, X_test, y_test, y_pred_test = stock_forecast.train_xgboost_model(symbol)
                stock_forecast.plot_model_results(df, X_test, y_test, y_pred_test)
                print(f"Bot: Analysis and prediction for {symbol} completed successfully!")
            else:
                print("Bot: Please set the stock data first.")
        
        elif intent == "greeting":
            print("Bot: Hello! How can I assist you with your stocks today?")
        
        elif intent == "thank_you":
            print("Bot: You're welcome! Let me know if you need any more help.")
        
        elif intent == "good_bye":
            print("Bot: Goodbye! Come back anytime for stock updates or predictions.")
            break
        
        else:
            print("Bot: I'm sorry, I didn't understand that. Can you try rephrasing it?")


# 4. Start the Chatbot
if __name__ == "__main__":
    chatbot()
