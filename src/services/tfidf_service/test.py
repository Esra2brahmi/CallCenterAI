import requests

# URL de l'endpoint /predict
SVM_SERVICE_URL = "http://localhost:8000/predict"

def test_predict(text):
    """Teste l'endpoint /predict avec le texte donné via une requête POST."""
    try:
        # Préparer la requête POST avec le texte
        payload = {"text": text}
        headers = {"Content-Type": "application/json"}
        
        # Envoi de la requête
        print(f"Envoi de la requête POST à {SVM_SERVICE_URL} avec le texte: {text[:50]}...")
        response = requests.post(SVM_SERVICE_URL, json=payload, headers=headers)
        response.raise_for_status()  # Lève une exception si la requête échoue
        
        # Afficher la réponse
        print("Réponse de l'API:", response.json())
        return response.json()
    
    except requests.exceptions.HTTPError as http_err:
        print(f"Erreur HTTP: {http_err}")
        print(f"Code de statut: {response.status_code}")
        print(f"Détails: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du test: {e}")
    return None

if __name__ == "__main__":
    # Texte donné pour le test
    test_text = "mail please dear looks blacklisted receiving mails anymore sample attached thanks kind regards senior engineer"
    test_predict(test_text)
    
    # Test supplémentaire avec un autre texte (optionnel)
    additional_test_text = "I am having issues with my account, please help!"
    print("\nTest supplémentaire:")
    test_predict(additional_test_text)