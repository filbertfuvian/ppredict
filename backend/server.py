import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from flask import Flask, request, jsonify
import pickle
import os
import logging
import pandas as pd

# Flask API server
app = Flask(__name__)

# Load model and label encoders for class, fold, family
model_class = load_model('model_class.h5')
model_fold = load_model('model_fold.h5')
model_family = load_model('model_family.h5')
with open('label_encoder_cl.pkl', 'rb') as f:
    le_cl = pickle.load(f)
with open('label_encoder_cf.pkl', 'rb') as f:
    le_cf = pickle.load(f)
with open('label_encoder_fa.pkl', 'rb') as f:
    le_fa = pickle.load(f)
with open('model_params.pkl', 'rb') as f:
    model_params = pickle.load(f)

def preprocess_sequence(sequence):
    max_len = model_params['max_len_95th'] if ('max_len_95th' in model_params) else 512
    aa_to_int = model_params['aa_to_int'] if ('aa_to_int' in model_params) else {aa: i+1 for i, aa in enumerate('ACDEFGHIKLMNPQRSTVWY')}
    seq_int = [aa_to_int.get(aa, 0) for aa in sequence[:max_len]]
    seq_int += [0] * (max_len - len(seq_int))
    arr = np.array(seq_int, dtype=np.int32).reshape(1, max_len)
    return arr

def decode_label(le, idx):
    try:
        node_id = int(le.inverse_transform([idx])[0])
        return get_name(node_id)
    except Exception:
        return str(idx)

lookup = {}
def load_lookup():
    global lookup
    df_lookup = pd.read_csv('scop-des-latest.csv')
    lookup = {int(row.NODE_ID): row.NODE_NAME for _, row in df_lookup.iterrows()}

def get_name(node_id):
    return lookup.get(int(node_id), str(node_id))

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Received request: {data}")
        sequences = data.get('sequences', [])
        
        if not sequences:
            logging.warning('No sequences provided in request')
            return jsonify({'error': 'No sequences provided'}), 400
            
        processed_seqs = [preprocess_sequence(seq) for seq in sequences]
        batch = np.vstack(processed_seqs)
        
        predictions_class = model_class.predict(batch)
        predictions_fold = model_fold.predict(batch)
        predictions_family = model_family.predict(batch)
        logging.info(f"Predictions shape: {[p.shape for p in [predictions_class, predictions_fold, predictions_family]]}")
        
        results = []
        for i, seq in enumerate(sequences):
            class_id = int(np.argmax(predictions_class[i]))
            fold_id = int(np.argmax(predictions_fold[i]))
            family_id = int(np.argmax(predictions_family[i]))
            result = {
                'sequence': str(seq),
                'predictions': {
                    'class': {
                        'id': int(class_id),
                        'name': str(decode_label(le_cl, class_id)),
                        'probability': float(np.max(predictions_class[i]))
                    },
                    'fold': {
                        'id': int(fold_id),
                        'name': str(decode_label(le_cf, fold_id)),
                        'probability': float(np.max(predictions_fold[i]))
                    },
                    'family': {
                        'id': int(family_id),
                        'name': str(decode_label(le_fa, family_id)),
                        'probability': float(np.max(predictions_family[i]))
                    }
                }
            }
            results.append(result)
        logging.info(f"Returning results for {len(results)} sequences")
        logging.info(f"Response: {results}")
        return jsonify({'results': results})
    
    except Exception as e:
        logging.exception('Error during prediction')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists('model_class.h5'):
        raise FileNotFoundError("Model file 'model_class.h5' tidak ditemukan.")
    if not os.path.exists('model_fold.h5'):
        raise FileNotFoundError("Model file 'model_fold.h5' tidak ditemukan.")
    if not os.path.exists('model_family.h5'):
        raise FileNotFoundError("Model file 'model_family.h5' tidak ditemukan.")
    
    load_lookup()
    app.run(host='0.0.0.0', port=5000)
