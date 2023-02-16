import streamlit as st
import transformers
import torch
tokenizer = transformers.GPT2Tokenizer.from_pretrained('gpt2')
model = transformers.GPT2LMHeadModel.from_pretrained('gpt2')
max_seq_len = 300
model_path = "angen.bin"


def generate_text(input_text, device='cuda', max_len=300):
    pad_tok = tokenizer.encode(['<|pad|>'])[0]
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()

    input_ids = tokenizer.encode(input_text)
    mask = [1]*len(input_ids)

    padding_len = max_seq_len - len(input_ids)

    input_ids = input_ids  # + pad_tok*padding_len
    mask = mask + [0]*padding_len

    ids = torch.tensor(input_ids, dtype=torch.long).to(device).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).to(device).unsqueeze(0)

    # print(ids[0])
    sample_out = model.generate(ids, min_length=30, max_length=max_len, pad_token_id=pad_tok,
                                top_p=0.85, early_stopping=True, do_sample=True, num_beams=5, no_repeat_ngram_size=2, num_return_sequences=1)

    story = (tokenizer.decode(sample_out[0], skip_special_tokens=True))
    return story


st.header("AnGen")
inp = st.text_input("Enter Initial Prompts")
if st.button("Submit"):
    st.subheader("Generated Story:")
    with st.spinner(text="This may take a moment..."):
        story = generate_text(inp, device='cuda')

    st.write(story)
