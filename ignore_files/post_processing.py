import pandas as pd
import re

def read_csv(filename):
    df = pd.read_csv(filename)
    return df

def string_to_bool(df):
    df.replace({'Sim': True, 'Não': False}, inplace=True)
    return df

def categorize_location(location):
    if type(location) != str:
        return False
    
    location = location.lower()
    if "lobo superior e inferior" in location:
        return "Outros"
    
    elif "língula" in location:
        return "lobo superior esquerdo"
    
    elif("médio" in location):
        return "Lobo médio direito"
    
    elif("direito" in location) or ("direita" in location):
        if "lobo superior" in location or "ápice" in location:
            return "Lobo superior direito"
        elif "lobo inferior" in location or "base" in location or "basal" in location:
            return "Lobo inferior direito"
        else:
            return "Outros"    
        
    elif("esquerda" in location) or ("esquerdo" in location):
        if "lobo superior" in location:
            return "Lobo superior esquerdo"
        elif "lobo inferior" in location or "base" in location or "basal" in location:
            return "Lobo inferior esquerdo"
        else:
            return "Outros"    
    else:
        return "Outros"
    
def extract_size(text):
    if text == False:
        return None
    if type(text) == float:
        text = str(text)
    text = text.lower()
    # Primeiro tentar extrair o formato composto
    composed_match = re.search(r'(\d+(?:,\d+)?\s?x\s?\d+(?:,\d+)?\s?(?:x\s?\d+(?:,\d+)?)?\s?(?:cm|mm))', text)
    if composed_match:
        return composed_match.group(0)
    # Se não encontrar, tentar extrair o formato simples
    simple_match = re.search(r'(\d+(?:,\d+)?\s?(?:cm|mm))', text)
    if simple_match:
        return simple_match.group(0)
    # Se não encontrar nenhum dos dois, retornar None
    return None

def convert_diameter_to_mm(value):
    if pd.isna(value):
        return value

    if "cm" in value:
        value = value.replace("cm", "").strip()
        value = value.replace(",", ".")  # Substitui vírgulas por pontos
        # Se houver "x", significa que é uma dimensão múltipla
        if "x" in value:
            dimention = [float(v.strip()) * 10 for v in value.split("x")]
            avg_diameter = sum(dimention) / len(dimention)
            return f"{avg_diameter:.1f}"
        else:
            return f"{float(value) * 10:.1f}"
    elif "mm" in value:
        value = value.replace("mm", "").strip()
        value = value.replace(",", ".")  # Substitui vírgulas por pontos
        # Se já estiver em mm, converte para float e mantém
        if "x" in value:
            dimention = [float(v.strip()) for v in value.split("x")]
            avg_diameter = sum(dimention) / len(dimention)
            return f"{avg_diameter:.1f}"
        else:
            return f"{float(value):.1f}"
    else:
        # Caso o valor não contenha "cm" ou "mm", retorna como está
        return value

def structured_location(df):
    df['Localização do nódulo'] = df['Localização do nódulo'].apply(categorize_location)
    return df

def structured_size(df):
    df['Tamanho do nódulo'] = df['Tamanho do nódulo'].apply(extract_size)
    return df

def converted_size(df):
    df['Tamanho do nódulo'] = df['Tamanho do nódulo'].apply(convert_diameter_to_mm)
    return df

if __name__ == '__main__':

    input_filename = "ignore/post_processed.csv"
    output_filename = "ignore/results_structured_post_processed.csv"

    df = read_csv(input_filename)
    df = string_to_bool(df)
    df = structured_location(df)
    df = structured_size(df)
    df = converted_size(df)
    df = df.rename(columns={"Tamanho do nódulo": "Tamanho do nódulo (mm)"})


    df.to_csv(output_filename, index=False)