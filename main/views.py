import requests
from django.shortcuts import render, redirect
from django.views.decorators.csrf import csrf_exempt
import random

def home(request):
    return render(request, 'home.html')

@csrf_exempt
def predict(request):
    if request.method == 'POST':
        sequence = request.POST.get('sequence', '').strip()
        if not sequence:
            return render(request, 'home.html', {'error': 'Please enter a protein sequence.'})
        try:
            response = requests.post(
                'http://192.168.0.104:5000/predict',
                json={'sequences': [sequence]},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            result = data['results'][0]
            pred_class = result['predictions']['class']
            pred_fold = result['predictions']['fold']
            pred_family = result['predictions']['family']
            context = {
                'protein_name': result['sequence'][:20] + ('...' if len(result['sequence']) > 20 else ''),
                'class': pred_class.get('name', pred_class),
                'fold': pred_fold.get('name', pred_fold),
                'family': pred_family.get('name', pred_family),
                'confidence': f"{max(pred_class['probability'], pred_fold['probability'], pred_family['probability']):.2f}",
                'top_scores': [
                    {'class': 'Class', 'name': pred_class.get('name', pred_class), 'score': pred_class['probability'] * 100},
                    {'class': 'Fold', 'name': pred_fold.get('name', pred_fold), 'score': pred_fold['probability'] * 100},
                    {'class': 'Family', 'name': pred_family.get('name', pred_family), 'score': pred_family['probability'] * 100},
                ],
            }
            return render(request, 'result.html', context)
        except Exception as e:
            return render(request, 'home.html', {'error': f'Prediction failed: {e}'})
    return redirect('home')

