import docx
import json
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pdb

tokenizer = AutoTokenizer.from_pretrained("Jean-Baptiste/camembert-ner")
ner_model = AutoModelForTokenClassification.from_pretrained("Jean-Baptiste/camembert-ner")
nlp = pipeline('ner', model=ner_model, tokenizer=tokenizer, aggregation_strategy="simple")

configuration = {"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": "en_core_web_sm" }]}
provider = NlpEngineProvider(nlp_configuration=configuration)
nlp_engine = provider.create_engine()
analyzer_engine = AnalyzerEngine(
    nlp_engine=nlp_engine, 
    supported_languages = ['en']
)

file_path = "AML Policy_Vault AM_v1_comments MvK.docx"
output_file = "detected_entities_spacy.jsonl"

try:
    document = docx.Document(file_path)
except Exception as e:
    print(f"Error loading the document: {e}")
    exit(1)

with open(output_file, "w", encoding="utf-8") as jsonl_file:
    for paragraph in document.paragraphs:
        if not paragraph.text.strip():
            continue

        original_text = paragraph.text

        try:
            # transformer_res = nlp(original_text)
            # detected_entities = [res['word'] for res in transformer_res]
            # print(set(detected_entities))
            results = analyzer_engine.analyze(
                text=original_text,
                entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "ORGANIZATION"],
                language="en"
            )
            if results:
                detected_entities = {}
                for result in results:
                    entity_type = result.entity_type
                    entity_value = original_text[result.start:result.end],
                    if entity_type not in detected_entities:
                        detected_entities[entity_type] = []
                    detected_entities[entity_type].append(entity_value)

                # Prepare JSON object
                json_object = {
                    "text": original_text,
                    "Entities": detected_entities
                }

                # Write JSON object to the JSONL file
                jsonl_file.write(json.dumps(json_object) + "\n")

        except Exception as e:
            print(f"Error analyzing text: {e}")

# Debugging complete
print("Processing complete.")