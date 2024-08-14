import re
def custom_tokenizer(text):
    pattern = r'\b[\w\.]+(?:/\d{2}){0,2}\b'
    tokens = re.findall(pattern, text)
    return tokens

text = 'Just received my M.Sc. diploma today, on 2024/02/10! Excited to embark on this new journey of knowledge and discovery. #MScGraduate #EducationMatters.'
tokens = custom_tokenizer(text)
print(tokens)