import pandas as pd


df = pd.read_csv('ignore/result_df_model.csv')

# Dicionário para armazenar as informações extraídas
nodule_info = {
    'Nódulo': [],
    'O nódulo é sólido?': [],
    'O nódulo é em partes moles, semissólido ou subsólido?': [],
    'O nódulo é em vidro fosco?': [],
    'O nódulo é espiculado, irregular ou mal definido?': [],
    'O nódulo é calcificado?': [],
    'Localização do nódulo': [],
    'Tamanho do nódulo (mm)': []
}

# Lista de palavras relacionadas às diferentes atenuações
atenuacoes_solido = ['sólido', 'sólidos']
atenuacoes_semissolido = ['subsólido', 'semissólido', 'semissólida', 'densidade de partes moles', 'densidade partes moles', 'atenuação de partes moles']
atenuacoes_vidro_fosco = ['atenuação com vidro fosco', 'atenuação em vidro fosco', 'totalmente em vidro fosco', 'em vidro fosco']
calcificacoes = ['calcificado', 'calcificados', 'hiperdensa', 'hiperdensas']

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
    
    # Variáveis auxiliares para construir entidades compostas
    current_entity = None
    current_value = []
    
    for _, row in report_df.iterrows():
        word = row['word']
        tag_pred = row['tag_pred']
        
        if tag_pred.startswith('B-'):
            # Quando começa uma nova entidade, armazenamos a anterior
            if current_entity:
                if current_entity == 'LOC':
                    nodule_data['location'].append(" ".join(current_value))
                elif current_entity == 'TAM':
                    nodule_data['size'].append("".join(current_value))
                    
            # Iniciar uma nova entidade
            current_entity = tag_pred.split('-')[1]
            current_value = [word]
            
        elif tag_pred.startswith('I-') and current_entity == tag_pred.split('-')[1]:
            # Continuar a entidade atual
            current_value.append(word)
        
        if tag_pred == 'B-ACH':
            lower_word = word.lower()
            if lower_word in atenuacoes_solido:
                nodule_data['solid'] = "Sim"
            elif lower_word in atenuacoes_semissolido:
                nodule_data['semisolid'] = "Sim"
            elif lower_word in atenuacoes_vidro_fosco:
                nodule_data['glass'] = "Sim"
        elif tag_pred == 'B-CAL':
            lower_word = word.lower()
            if lower_word in calcificacoes:
                nodule_data['calcified'] = "Sim"

    # Armazenar a última entidade processada
    if current_entity:
        if current_entity == 'LOC':
            nodule_data['location'].append(" ".join(current_value))
        elif current_entity == 'TAM':
            nodule_data['size'].append("".join(current_value))
    
    return nodule_data

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
    nodule_info['Tamanho do nódulo (mm)'].append(", ".join(nodule_data['size']) if nodule_data['size'] else None)


# Criar DataFrame final
df_final = pd.DataFrame(nodule_info)
df_final.to_csv('ignore/post_processed.csv', index=False)
