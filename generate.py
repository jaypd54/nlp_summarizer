from transformers import T5Tokenizer, T5ForConditionalGeneration

model_path = "model/t5_summarizer_model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

def generate_summary(text):
    input_text = "summarize: " + text

    inputs = tokenizer(
        input_text,
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=80,
        min_length=20,
        num_beams=6,
        repetition_penalty=1.5,
        no_repeat_ngram_size=3,
        early_stopping=True
    )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)