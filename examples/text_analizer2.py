import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
tf.compat.v1.disable_eager_execution()

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertModel.from_pretrained('bert-base-uncased')

def get_bert_similarity(sent1, sent2):
    # Tokenize the two sentences
    encoded_sent1 = tokenizer.encode_plus(sent1, add_special_tokens=True, return_tensors='tf')
    encoded_sent2 = tokenizer.encode_plus(sent2, add_special_tokens=True, return_tensors='tf')

    # Get the input IDs and attention masks for each sentence
    input_ids_sent1 = encoded_sent1['input_ids']
    attention_mask_sent1 = encoded_sent1['attention_mask']
    input_ids_sent2 = encoded_sent2['input_ids']
    attention_mask_sent2 = encoded_sent2['attention_mask']


    # Pass the input IDs and attention masks through the BERT model
    outputs_sent1 = model(input_ids_sent1, attention_mask=attention_mask_sent1)
    outputs_sent2 = model(input_ids_sent2, attention_mask=attention_mask_sent2)

    # Extract the embeddings for each sentence from the output of the BERT model
    embeddings_sent1 = outputs_sent1.last_hidden_state[:, 0, :]
    embeddings_sent2 = outputs_sent2.last_hidden_state[:, 0, :]

    # Calculate the cosine similarity between the two embeddings
    similarity = tf.keras.losses.cosine_similarity(embeddings_sent1, embeddings_sent2).numpy()[0][0]

    return similarity

sent1 = 'I like cats'
sent2 = 'I like dogs'
similarity = get_bert_similarity(sent1, sent2)
print(similarity)