import pandas as pd
import re

def read_csv(filename):
    df = pd.read_csv(filename)
    return df

# Função de mapeamento com base nas entidades e palavras do relatório
def extract_info_from_report(report_df):
    nodule_data = {
        'solid': "Não",
        'semisolid': "Não",
        'glass': "Não",
        'irregular': "Não",
        'calcified': "Não",
        'location': [],
        'size': []
    }
    
    # Lista de palavras relacionadas às diferentes atenuações
    atenuacoes_solido = ['sólido', 'sólidos']
    atenuacoes_semissolido = ['subsólido', 'semissólido', 'semissólida', 'densidade de partes moles', 'densidade partes moles', 'atenuação de partes moles']
    atenuacoes_vidro_fosco = ['atenuação com vidro fosco', 'atenuação em vidro fosco', 'totalmente em vidro fosco', 'em vidro fosco']
    calcificacoes = ['calcificado', 'calcificados', 'hiperdensa', 'hiperdensas']
    bordas = ['contornos irregulares', 'irregulares', 'margens espiculadas', 'espiculadas']
    
    # Variáveis auxiliares para construir entidades compostas
    current_entity = None
    current_value = []
    
    for _, row in report_df.iterrows():
        Token = row['word']
        tag_pred = row['tag_pred']
        
        # Tratamento para novas entidades com 'B-'
        if tag_pred.startswith('B-'):
            # Armazenar a entidade anterior
            if current_entity:
                if current_entity == 'LOC':
                    nodule_data['location'].append(" ".join(current_value))
                elif current_entity == 'TAM':
                    nodule_data['size'].append("".join(current_value))
                    
            # Iniciar uma nova entidade
            current_entity = tag_pred.split('-')[1]
            current_value = [Token]
        
        # Tratamento para continuidade de entidades com 'I-'
        elif tag_pred.startswith('I-') and current_entity == tag_pred.split('-')[1]:
            current_value.append(Token)
        
        # Verificação para atributos específicos de acordo com 'B-' e 'I-' em 'ACH', 'CAL', 'BOR'
        entity_text = " ".join(current_value).lower()
        
        if current_entity == 'ATE':
            if entity_text  in atenuacoes_solido:
                nodule_data['solid'] = "Sim"
            elif entity_text  in atenuacoes_semissolido:
                nodule_data['semisolid'] = "Sim"
            elif entity_text  in atenuacoes_vidro_fosco:
                nodule_data['glass'] = "Sim"
        
        elif current_entity == 'CAL':
            if entity_text in calcificacoes:
                nodule_data['calcified'] = "Sim"
        
        elif current_entity == 'BOR':
            if entity_text  in bordas:
                nodule_data['irregular'] = "Sim"

    # Armazenar a última entidade processada
    if current_entity:
        if current_entity == 'LOC':
            nodule_data['location'].append(" ".join(current_value))
        elif current_entity == 'TAM':
            nodule_data['size'].append("".join(current_value))
    
    return nodule_data

def create_dataframe_post_processed(df, nodule_info):
    # Agrupar os dados por relatório e processar cada um
    for report_id, group in df.groupby('report'):
        nodule_data = extract_info_from_report(group)
        nodule_info['Nódulo'].append(report_id)
        nodule_info['O nódulo é sólido?'].append(nodule_data['solid'])
        nodule_info['O nódulo é em partes moles, semissólido ou subsólido?'].append(nodule_data['semisolid'])
        nodule_info['O nódulo é em vidro fosco?'].append(nodule_data['glass'])
        nodule_info['O nódulo é espiculado, irregular ou mal definido?'].append(nodule_data['irregular'])
        nodule_info['O nódulo é calcificado?'].append(nodule_data['calcified'])
        nodule_info['Localização do nódulo'].append(", ".join(nodule_data['location']) if nodule_data['location'] else None)
        nodule_info['Tamanho do nódulo'].append(", ".join(nodule_data['size']) if nodule_data['size'] else None)

    # Criar DataFrame final
    df_final = pd.DataFrame(nodule_info)
    #df_final.to_csv('data/results_post_processed.csv', index=False)
    return df_final


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
    if text == None:
        return False
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

    if value == False:
        return value

    if "cm" in value:
        value = value.replace("cm", "").strip()
        value = value.replace(",", ".")  # Substitui vírgulas por pontos
        # Se houver "x", significa que é uma dimensão múltipla
        if "x" in value:
            dimention = [float(v.strip()) * 10 for v in value.split("x")]
            #avg_diameter = sum(dimention) / len(dimention)
            min_diameter = min(dimention)
            #return f"{avg_diameter:.1f}"
            return f"{min_diameter:.1f}"
        else:
            return f"{float(value) * 10:.1f}"
    elif "mm" in value:
        value = value.replace("mm", "").strip()
        value = value.replace(",", ".")  # Substitui vírgulas por pontos
        # Se já estiver em mm, converte para float e mantém
        if "x" in value:
            dimention = [float(v.strip()) for v in value.split("x")]
            #avg_diameter = sum(dimention) / len(dimention)
            min_diameter = min(dimention)
            #return f"{avg_diameter:.1f}"
            return f"{min_diameter:.1f}"
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
    
    df_result = pd.read_csv('data/result_df_model.csv')

    # Dicionário para armazenar as informações extraídas
    nodule_info = {
        'Nódulo': [],
        'O nódulo é sólido?': [],
        'O nódulo é em partes moles, semissólido ou subsólido?': [],
        'O nódulo é em vidro fosco?': [],
        'O nódulo é espiculado, irregular ou mal definido?': [],
        'O nódulo é calcificado?': [],
        'Localização do nódulo': [],
        'Tamanho do nódulo': []
    }

    df_results_post_processed = create_dataframe_post_processed(df_result, nodule_info)
    df_results_post_processed = string_to_bool(df_results_post_processed)
    df_results_post_processed = structured_location(df_results_post_processed)
    df_results_post_processed = structured_size(df_results_post_processed)
    df_results_post_processed = converted_size(df_results_post_processed)
    df_results_post_processed = df_results_post_processed.rename(columns={"Tamanho do nódulo": "Tamanho do nódulo (mm)"})
    df_results_post_processed.to_csv('data/bilstmcrf_results_post_processed.csv', encoding='utf-8', index=False)


