import PyPDF2
import spacy
from transformers import pipeline
import tkinter as tk
from tkinter import filedialog
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# PDF dosyasını okuma
def pdf_oku(dosya_adı):
    with open(dosya_adı, 'rb') as file:
        pdf_reader = PyPDF2.PdfFileReader(file)
        metin = ""
        for sayfa_num in range(pdf_reader.numPages):
            sayfa = pdf_reader.getPage(sayfa_num)
            metin += sayfa.extractText()
        return metin

# SpaCy dil modelini yükleyin
nlp = spacy.load("en_core_web_sm")

# Soru sorma ve cevap alma işlemleri
def soru_cevap(pdf_metin, soru):
    # SpaCy ile belgedeki anahtar kelimeleri bulma
    belge = nlp(pdf_metin)
    anahtar_kelimeler = [kelime.text for kelime in belge if not kelime.is_stop and kelime.is_alpha][:10]

    # Hugging Face Transformers ile önceden eğitilmiş QA modelini kullanma
    model = pipeline('question-answering', model='distilbert-base-cased-distilled-squad', tokenizer='distilbert-base-cased')
    cevap = model(question=soru, context=pdf_metin)

    # Anahtar kelimelerle soruyu karşılaştırma
    eslesen_kelimeler = set(anahtar_kelimeler) & set([kelime.text for kelime in nlp(soru) if not kelime.is_stop and kelime.is_alpha])

    if eslesen_kelimeler:
        return "Evet, belgede bu konu hakkında bilgi bulunuyor. Cevap: " + cevap['answer']
    else:
        return "Üzgünüm, belgede bu konu hakkında bilgi bulunmuyor."

# Dosya seçme işlemi
def dosya_sec():
    dosya_adı = filedialog.askopenfilename()
    if dosya_adı:
        pdf_metin = pdf_oku(dosya_adı)
        soru = input("Sormak istediğiniz bir soru girin: ")
        cevap = soru_cevap(pdf_metin, soru)
        print("Cevap: " + cevap)

# Ana işlem
root = tk.Tk()
root.withdraw()

print("Lütfen PDF dosyasını seçin:")
dosya_sec()
