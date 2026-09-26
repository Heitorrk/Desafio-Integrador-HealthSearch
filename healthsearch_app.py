import math
import re
from typing import List, Dict, Any
import io

import streamlit as st
import pandas as pd
import numpy as np
import pypdf

import nltk
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder, util

# -----------------------------------------------------------------------------
# DOWNLOAD DE RECURSOS DO NLTK
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def download_nltk_resources():
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)

download_nltk_resources()

# -----------------------------------------------------------------------------
# CONFIGURAÇÃO DA PÁGINA STREAMLIT
# -----------------------------------------------------------------------------
st.set_page_config(page_title="HealthSearch - Motor Híbrido", page_icon="🏥", layout="wide")
st.title("HealthSearch — Motor de Busca Híbrido")
st.markdown("Busca inteligente médica integrando **BM25** (Léxico) e **Embeddings** (Semântico) via **Reciprocal Rank Fusion (RRF)**.")

# -----------------------------------------------------------------------------
# FASE 1: INGESTÃO DO CORPUS E PRÉ-PROCESSAMENTO
# -----------------------------------------------------------------------------
uploaded_files = st.file_uploader("Envie arquivos PDF para usar como base de documentos (substitui a base padrão)", type=["pdf"], accept_multiple_files=True)

documents_data = []

if uploaded_files:
    for file in uploaded_files:
        try:
            reader = pypdf.PdfReader(io.BytesIO(file.read()))
            texto = " ".join([page.extract_text() for page in reader.pages if page.extract_text()])
            if texto.strip():
                documents_data.append({
                    "ID": f"PDF_{len(documents_data)+1}",
                    "Título": file.name,
                    "Texto": texto
                })
        except Exception as e:
            st.error(f"Erro ao processar {file.name}: {e}")

if not documents_data:
    # BASE DE DADOS DO CORPUS MÉDICO (HARDCODE OBRIGATÓRIO, MAS COM TEXTOS LONGOS)
    documents_data = [
{"ID": "Doc 1", "Título": "Protocolo Emergência ECG", "Texto": "Pacientes com dor precordial aguda e suspeita de síndrome coronariana aguda devem realizar eletrocardiograma (CÓD-ECG-12D) em até 10 minutos após a admissão na unidade de emergência. A avaliação inicial também deve incluir o monitoramento contínuo dos sinais vitais, a administração de oxigênio suplementar se a saturação estiver abaixo de 90% e a coleta imediata de marcadores de necrose miocárdica (como troponina I ou T de alta sensibilidade). Em casos de elevação do segmento ST, a estratégia de reperfusão, preferencialmente intervenção coronária percutânea (angioplastia) primária, deve ser acionada prontamente, visando um tempo porta-balão inferior a 90 minutos para minimizar o dano ao músculo cardíaco. Além disso, é fundamental que a equipe de enfermagem esteja treinada para reconhecer os padrões eletrocardiográficos de isquemia aguda, como inversão de onda T, supra ou infra de ST, e bloqueios de ramo, que podem mascarar o diagnóstico. A realização do ECG deve ser repetida em intervalos de 15 a 30 minutos se os sintomas persistirem ou evoluírem, e a comparação com traçados anteriores é essencial para detectar alterações dinâmicas. O paciente deve ser mantido em repouso absoluto, com acesso venoso periférico calibroso já estabelecido, e deve-se evitar a administração de nitratos sublinguais se houver suspeita de infarto de ventrículo direito ou uso recente de inibidores da fosfodiesterase. A analgesia com morfina endovenosa pode ser considerada para alívio da dor refratária, mas com cautela devido ao risco de hipotensão e depressão respiratória. A estratificação de risco deve ser feita continuamente, utilizando escores como o GRACE ou o TIMI, para guiar a decisão entre estratégia invasiva precoce ou conservadora. Em hospitais sem capacidade para angioplastia primária, a fibrinólise deve ser iniciada em até 30 minutos da chegada, se não houver contraindicações, e o paciente deve ser transferido para um centro de referência imediatamente após a infusão do trombolítico. A monitorização da dor, da frequência cardíaca, da pressão arterial e da saturação de oxigênio deve ser contínua, e a equipe deve estar atenta a arritmias malignas, como taquicardia ventricular e fibrilação ventricular, que podem surgir nas primeiras horas do evento. A comunicação com a equipe de hemodinâmica deve ser ágil e eficiente, com envio antecipado do laudo eletrocardiográfico e dos dados clínicos para agilizar o preparo da sala de cateterismo. Por fim, a orientação ao paciente e familiares sobre o quadro, os procedimentos e os riscos envolvidos é parte integrante do cuidado humanizado, devendo ser documentada em prontuário, assim como todos os tempos de resposta e intervenções realizadas, para fins de auditoria e melhoria contínua da qualidade assistencial."},
{"ID": "Doc 2", "Título": "Guia de Farmacologia Cardíaca", "Texto": "O uso imediato de ácido acetilsalicílico e antiagregantes plaquetários reduz substancialmente a mortalidade no infarto agudo do miocárdio. Em pacientes sem contraindicações absolutas (como sangramento gastrointestinal ativo ou alergia severa), o AAS deve ser administrado na dose de 160 a 325 mg mastigado o mais precocemente possível. Ademais, a terapia dupla com inibidores do receptor P2Y12 (como clopidogrel, ticagrelor ou prasugrel) é fundamental no manejo conservador ou invasivo, sendo a duração do tratamento tipicamente recomendada por pelo menos 12 meses, juntamente com estatinas de alta intensidade para estabilização de placas ateroscleróticas e betabloqueadores para reduzir a demanda de oxigênio do miocárdio. A escolha do agente antiplaquetário deve considerar o perfil de risco do paciente, incluindo idade, função renal, histórico de acidente vascular cerebral prévio e risco de sangramento, sendo que o ticagrelor é preferido em pacientes de alto risco isquêmico devido à sua ação mais rápida e reversível, enquanto o prasugrel é reservado para pacientes sem histórico de AVC ou TIA e com menos de 75 anos. O clopidogrel permanece uma opção válida em casos de custo-efetividade ou quando os demais agentes não estão disponíveis. Além dos antiagregantes, a administração de nitroglicerina sublingual ou endovenosa é indicada para alívio da dor anginosa e redução da pré-carga, mas deve ser usada com cautela em pacientes com infarto de ventrículo direito ou hipotensão significativa. Os betabloqueadores, como metoprolol ou carvedilol, devem ser iniciados precocemente, desde que não haja sinais de insuficiência cardíaca aguda, bradicardia severa ou bloqueio atrioventricular de alto grau, e a dose deve ser titulada cuidadosamente para atingir frequência cardíaca entre 50 e 60 batimentos por minuto. Os inibidores da enzima conversora de angiotensina (IECA) ou os bloqueadores dos receptores da angiotensina II (BRA) são recomendados nas primeiras 24 horas para pacientes com disfunção sistólica, hipertensão ou diabetes, contribuindo para o remodelamento ventricular favorável. As estatinas de alta potência, como atorvastatina 80 mg ou rosuvastatina 40 mg, devem ser iniciadas imediatamente, independentemente do nível basal de colesterol, pois seus efeitos pleiotrópicos incluem estabilização da placa, redução da inflamação e melhora da função endotelial. O manejo farmacológico também envolve a correção agressiva de distúrbios eletrolíticos, especialmente hipocalemia e hipomagnesemia, que predispõem a arritmias fatais, e a monitorização da função renal e hepática é obrigatória durante a internação. A interação medicamentosa deve ser constantemente revisada, especialmente com anticoagulantes, anti-inflamatórios não esteroides e outros agentes que afetam a hemostasia. A educação do paciente sobre a adesão ao tratamento, os sinais de alerta de eventos adversos e a importância do seguimento ambulatorial é tão crucial quanto a prescrição correta, e a farmácia hospitalar deve garantir o fornecimento ininterrupto de todos os medicamentos, com checagem de validade e armazenamento adequado."},
{"ID": "Doc 3", "Título": "Diretriz de Hipertensão Arterial", "Texto": "A crise hipertensiva severa, configurando emergência hipertensiva devido a sinais de lesão de órgão-alvo agudo e progressivo, requer administração de anti-hipertensivos venosos e monitoramento contínuo da pressão arterial em Unidade de Terapia Intensiva (UTI). Os agentes de escolha frequentemente incluem o nitroprussiato de sódio, a nitroglicerina (especialmente em síndromes coronarianas) e betabloqueadores venosos como o esmolol. A redução da pressão arterial deve ser gradual, visando uma diminuição inicial não superior a 25% na primeira hora, seguida por uma redução lenta e cautelosa nas 23 horas subsequentes, a fim de prevenir a isquemia cerebral, renal e miocárdica que poderia decorrer de uma hipotensão abrupta e excessiva. O nitroprussiato é um potente vasodilatador arterial e venoso, com início de ação em segundos e meia-vida curta, permitindo titulação fina, mas seu uso prolongado ou em altas doses pode levar a toxicidade por cianeto, especialmente em pacientes com insuficiência renal, exigindo monitorização dos níveis de lactato e pH arterial. A nitroglicerina venosa é preferível em pacientes com insuficiência coronária associada, pois reduz a pré-carga e alivia a isquemia miocárdica, mas pode causar cefaleia intensa e taquifilaxia com o tempo. O esmolol é indicado em dissecção aórtica aguda ou taquicardia reflexa, com infusão contínua e ajuste rigoroso da frequência cardíaca. Outros agentes como o clevidipino, um bloqueador de canais de cálcio dihidropiridínico, têm ganhado espaço devido ao seu perfil de segurança e controle rápido da pressão, mas ainda não estão disponíveis em todos os centros. A monitorização invasiva da pressão arterial por cateter intra-arterial é recomendada para pacientes instáveis ou em uso de agentes vasoativos potentes, permitindo leituras em tempo real e reduzindo erros de aferição. Paralelamente, deve-se investigar a causa subjacente da emergência, como feocromocitoma, eclâmpsia, uso de drogas ilícitas (cocaína, anfetaminas) ou interrupção abrupta de anti-hipertensivos crônicos, para direcionar o tratamento etiológico. A reposição volêmica deve ser feita com cautela, pois muitos pacientes já apresentam sobrecarga hídrica, e o uso de diuréticos de alça pode ser necessário se houver edema pulmonar ou insuficiência cardíaca congestiva. A equipe multidisciplinar, incluindo nefrologista e cardiologista, deve ser envolvida precocemente para otimizar a terapia e planejar a transição para anti-hipertensivos orais assim que a pressão estiver controlada e o paciente estável. A documentação detalhada de todos os parâmetros vitais, doses administradas, respostas clínicas e eventos adversos é obrigatória, e a família deve ser informada sobre a gravidade do quadro e as etapas do tratamento, garantindo consentimento informado para procedimentos invasivos, se necessários."},
{"ID": "Doc 4", "Título": "Manual de AVC Isquêmico", "Texto": "O acidente vascular cerebral (AVC) isquêmico agudo deve ser tratado com trombolíticos venosos, especificamente o ativador do plasminogênio tecidual recombinante (rt-PA), em até quatro horas e meia do início dos sintomas focais, desde que preenchidos os rigorosos critérios de elegibilidade. O manejo inicial exige a rápida realização de uma tomografia computadorizada (TC) de crânio sem contraste para descartar quadros hemorrágicos antes de qualquer intervenção fibrinolítica. Em centros especializados, pacientes que apresentam oclusão de grandes vasos (LVO) podem se beneficiar da terapia de trombectomia mecânica endovascular, que pode ser realizada num prazo ampliado de até 24 horas caso critérios avançados de neuroimagem (como mismatch na ressonância ou TC perfusão) sejam atendidos satisfatoriamente. A avaliação neurológica seriada, utilizando a escala NIHSS, é indispensável para monitorar a evolução clínica e detectar precocemente deterioração neurológica, que pode indicar expansão do infarto, hemorragia transformada ou edema cerebral maligno. A pressão arterial deve ser mantida em níveis controlados, geralmente abaixo de 185/110 mmHg antes da trombólise e abaixo de 180/105 mmHg nas primeiras 24 horas após o tratamento, para reduzir o risco de sangramento intracraniano. A glicemia capilar deve ser verificada imediatamente, corrigindo-se hipoglicemia ou hiperglicemia severa, pois ambas estão associadas a piores desfechos funcionais. A disfagia deve ser rastreada antes da administração de qualquer dieta ou medicação oral, e a nutrição enteral precoce é indicada para pacientes com comprometimento do nível de consciência ou incapacidade de deglutir. A profilaxia para trombose venosa profunda com compressão pneumática intermitente e heparina subcutânea em baixas doses deve ser iniciada precocemente, assim como a mobilização passiva e ativa precoce, supervisionada por fisioterapeuta, para prevenir complicações musculoesqueléticas e tromboembólicas. O manejo da hipertermia é crítico, pois temperaturas acima de 37,5°C pioram o dano neuronal, devendo-se investigar e tratar infecções subjacentes, como pneumonia ou infecção urinária, com antibioticoterapia adequada. A equipe de enfermagem desempenha papel central na monitorização contínua da pressão arterial, frequência cardíaca, saturação de oxigênio e estado neurológico, além de garantir a administração correta e pontual dos medicamentos, evitando interrupções desnecessárias. A comunicação com a família deve ser clara e empática, explicando a janela terapêutica, os riscos e benefícios dos procedimentos, e a importância da reabilitação precoce, que inclui fonoaudiologia, terapia ocupacional e psicologia, para maximizar a recuperação funcional e a qualidade de vida após a alta hospitalar."},
{"ID": "Doc 5", "Título": "Protocolo de Reanimação RCR", "Texto": "A parada cardiorrespiratória (PCR) em adultos exige o acionamento imediato do protocolo de código azul, seguido por compressões torácicas contínuas de alta qualidade, minimizando interrupções, garantindo uma profundidade de pelo menos 5 centímetros (sem ultrapassar 6 cm) e mantendo uma frequência entre 100 e 120 compressões por minuto. A desfibrilação precoce é o passo mais crítico para ritmos chocáveis como a Fibrilação Ventricular (FV) ou a Taquicardia Ventricular Sem Pulso (TVSP), com o primeiro choque administrado preferencialmente nos primeiros três a cinco minutos após o colapso. O manejo das vias aéreas com ventilação adequada e o acesso venoso ou intraósseo para a administração de epinefrina (1 mg a cada 3 a 5 minutos) completam o suporte avançado de vida. As compressões devem ser realizadas com o paciente em superfície rígida, com as mãos posicionadas no centro do tórax, sobre o terço inferior do esterno, e o socorrista deve permitir o retorno completo do tórax entre as compressões, sem se apoiar no peito, para maximizar o retorno venoso e o débito cardíaco. O uso de dispositivos mecânicos de compressão pode ser considerado em situações de transporte ou quando a reanimação manual é difícil, mas não substituem a qualidade das compressões manuais bem executadas. A ventilação deve ser feita com relação compressão-ventilação de 30:2 para um único socorrista, ou com via aérea avançada (tubo endotraqueal ou máscara laríngea) com ventilação contínua a cada 6 segundos, sem interromper as compressões. A capnografia quantitativa (EtCO2) é fortemente recomendada para confirmar a posição do tubo, monitorar a qualidade da RCP (valores > 10 mmHg indicam compressões eficazes) e detectar o retorno da circulação espontânea (ROSC), que é marcado por um aumento súbito do EtCO2. A administração de amiodarona (300 mg em bolo, seguido de 150 mg) ou lidocaína é indicada para FV/TVSP refratária ao choque e epinefrina, enquanto o sulfato de magnésio é reservado para torsades de pointes ou hipomagnesemia confirmada. Após a ROSC, o manejo pós-parada inclui controle rigoroso da temperatura corporal alvo (hipotermia terapêutica entre 32°C e 36°C) por pelo menos 24 horas, monitorização hemodinâmica invasiva, correção de distúrbios eletrolíticos e ácido-base, e avaliação neurológica precoce com EEG contínuo se houver suspeita de convulsões subclínicas. A equipe deve realizar debriefing após cada evento de PCR para revisar o desempenho, identificar falhas no processo e implementar melhorias no treinamento e na logística do código azul, garantindo que todos os membros estejam atualizados com as diretrizes mais recentes da American Heart Association ou do European Resuscitation Council, e que os equipamentos (desfibriladores, carrinhos de emergência, medicamentos) sejam checados diariamente e estejam prontos para uso imediato."},
{"ID": "Doc 6", "Título": "Procedimentos de UTI Geral", "Texto": "Para diagnóstico do protocolo CÓD-ECG-12D em arritmias complexas ou em pacientes com alto risco de degeneração hemodinâmica, recomenda-se fortemente a monitorização cardíaca contínua por telemetria na Unidade de Terapia Intensiva (UTI). Esses procedimentos de terapia intensiva geral envolvem a observação ininterrupta dos traçados de ECG, bem como a avaliação seriada da saturação arterial de oxigênio por oximetria de pulso, medição invasiva ou não invasiva da pressão arterial, e controle rigoroso do balanço hídrico e de eletrólitos (como potássio e magnésio), que são cruciais para a manutenção da estabilidade elétrica do miocárdio e para prevenir eventos adversos letais decorrentes do prolongamento do intervalo QT ou bloqueios atrioventriculares de alto grau. A monitorização hemodinâmica avançada, com cateter de artéria pulmonar ou dispositivos menos invasivos como o Vigileo ou o PiCCO, pode ser necessária para pacientes em choque cardiogênico, séptico ou com insuficiência respiratória grave, permitindo a otimização da pré-carga, pós-carga e contratilidade, com base em parâmetros como débito cardíaco, índice cardíaco, resistência vascular sistêmica e pressão de oclusão da artéria pulmonar. A ventilação mecânica invasiva é frequente na UTI, e o ajuste dos parâmetros ventilatórios (volume corrente, pressão expiratória final positiva - PEEP, fração inspirada de oxigênio - FiO2) deve ser individualizado, com monitorização da mecânica pulmonar e gasometria arterial seriada para evitar lesão pulmonar induzida pelo ventilador e garantir oxigenação e ventilação adequadas. A sedação e analgesia devem ser balanceadas, utilizando escalas validadas (RASS ou SAS) para manter o paciente confortável e cooperativo, mas evitando sedação excessiva que prolongue o tempo de desmame ventilatório. A profilaxia para úlcera de estresse, trombose venosa profunda e infecções relacionadas a cateteres é obrigatória, com protocolos de higiene das mãos, curativos estéreis e troca programada de dispositivos invasivos. A nutrição enteral precoce, iniciada nas primeiras 24 a 48 horas, é preferível à parenteral, desde que o trato gastrointestinal esteja funcionante, e a oferta calórico-proteica deve ser calculada com base no peso, estresse metabólico e função renal. A equipe multiprofissional, composta por intensivistas, enfermeiros, fisioterapeutas, nutricionistas e farmacêuticos, realiza rondas diárias para revisar cada caso, ajustar condutas, planejar a alta da UTI e discutir aspectos éticos, como diretivas antecipadas de vontade e limitação de suporte terapêutico, quando cabível. A documentação eletrônica em tempo real de todos os parâmetros, intervenções e evoluções é essencial para a continuidade do cuidado, auditoria interna e pesquisa clínica, e a comunicação com a família deve ser frequente e transparente, oferecendo suporte emocional e informações claras sobre o prognóstico e as opções terapêuticas disponíveis."}
]

def clean_and_tokenize(text: str) -> List[str]:
    """Limpa o texto, remove stopwords e tokeniza."""
    # Normalização para minúsculas e remoção de caracteres especiais, preservando hífens e números
    text = re.sub(r'[^\w\s\-]', '', text)
    text = text.lower().strip()
    
    tokens = text.split()
    try:
        stop_words = set(stopwords.words('portuguese'))
    except Exception:
        stop_words = set()
        
    return [t for t in tokens if t not in stop_words and len(t) > 1]

# Preparação dos dados lexicais
corpus_tokenized = [clean_and_tokenize(doc["Texto"]) for doc in documents_data]

# -----------------------------------------------------------------------------
# CARREGAMENTO DOS MODELOS (FASE 3 E BÔNUS)
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner="Carregando modelo de embeddings semânticos...")
def load_embedding_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

@st.cache_resource(show_spinner="Carregando modelo de Cross-Encoder (Re-Ranking)...")
def load_cross_encoder():
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

emb_model = load_embedding_model()

@st.cache_data(show_spinner="Calculando embeddings do corpus...")
def get_corpus_embeddings(texts: List[str]):
    return emb_model.encode(texts)

corpus_embeddings = get_corpus_embeddings([doc["Texto"] for doc in documents_data])

# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# PARÂMETROS E CONTROLES
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
# BARRA LATERAL (SIDEBAR)
# -----------------------------------------------------------------------------
st.sidebar.header("Motor Léxico (BM25)")
k1 = st.sidebar.slider("Parâmetro $k_1$ (Saturação de Frequência)", min_value=0.0, max_value=3.0, value=1.2, step=0.1)
b = st.sidebar.slider("Parâmetro $b$ (Normalização pelo Comprimento)", min_value=0.0, max_value=1.0, value=0.75, step=0.05)

st.sidebar.header("Algoritmo de Fusão (RRF)")
alpha = st.sidebar.slider("Peso α (BM25 vs Semântico)", min_value=0.0, max_value=1.0, value=0.5, step=0.1)
k_rrf = 60 # Constante da fórmula

st.sidebar.header("Desafio Bônus")
use_cross_encoder = st.sidebar.checkbox("Ativar Cross-Encoder Re-Ranking (Top-3)", value=False)

# -----------------------------------------------------------------------------
# BUSCA E PROCESSAMENTO
# -----------------------------------------------------------------------------
query = st.text_input("🔍 Digite sua consulta médica:", value="infarto")

if query:
    # 1. Busca Léxica (BM25)
    bm25 = BM25Okapi(corpus_tokenized, k1=k1, b=b)
    query_tokenized = clean_and_tokenize(query)
    bm25_scores = bm25.get_scores(query_tokenized)
    
    # 2. Busca Semântica Vetorial (Similaridade de Cosseno)
    query_emb = emb_model.encode([query])
    sem_scores = util.cos_sim(query_emb, corpus_embeddings)[0].numpy()
    
    # Gerando os Rankings para usar na fórmula RRF
    # Rank 1 é o melhor (maior score). argsort() retorna do menor para o maior, então invertemos [::-1]
    lexical_ranks = {idx: rank+1 for rank, idx in enumerate(np.argsort(bm25_scores)[::-1])}
    semantic_ranks = {idx: rank+1 for rank, idx in enumerate(np.argsort(sem_scores)[::-1])}
    
    results = []
    
    for idx, doc in enumerate(documents_data):
        rank_bm25 = lexical_ranks[idx]
        rank_sem = semantic_ranks[idx]
        score_bm25_raw = bm25_scores[idx]
        score_sem_raw = sem_scores[idx]
        
        # 3. Cálculo RRF
        rrf_score = alpha * (1 / (k_rrf + rank_bm25)) + (1 - alpha) * (1 / (k_rrf + rank_sem))
        
        results.append({
            "ID": doc["ID"],
            "Título": doc["Título"],
            "Texto": doc["Texto"],
            "Score BM25": float(score_bm25_raw),
            "Rank BM25": rank_bm25,
            "Score Semântico": float(score_sem_raw),
            "Rank Semântico": rank_sem,
            "Score RRF": float(rrf_score)
        })
    
    # Ordena resultados híbridos
    hybrid_results = sorted(results, key=lambda x: x["Score RRF"], reverse=True)
    
    # 4. Desafio Bônus: Cross-Encoder Re-Ranking nos Top 3
    if use_cross_encoder:
        ce_model = load_cross_encoder()
        top_3 = hybrid_results[:3]
        pairs = [[query, doc["Texto"]] for doc in top_3]
        ce_scores = ce_model.predict(pairs)
        
        for i, doc in enumerate(top_3):
            doc["Score Cross-Encoder"] = float(ce_scores[i])
            
        # Re-ordena apenas os Top 3 usando o score do Cross-Encoder
        hybrid_results[:3] = sorted(top_3, key=lambda x: x["Score Cross-Encoder"], reverse=True)
        # O resto mantém a ordem RRF
    
    # -----------------------------------------------------------------------------
    # INTERFACE & ABAS (STREAMLIT)
    # -----------------------------------------------------------------------------
    tab_lex, tab_sem, tab_hyb, tab_comp = st.tabs([
        "🔤 Busca Léxica (BM25)", 
        "🧠 Busca Semântica Vetorial", 
        "🔀 Híbrido RRF" + (" + Re-Rank" if use_cross_encoder else ""), 
        "📊 Matriz Comparativa"
    ])
    
    with tab_lex:
        st.subheader("Resultados do Motor Léxico (Okapi BM25)")
        df_lex = pd.DataFrame(results).sort_values(by="Score BM25", ascending=False)
        st.dataframe(df_lex[["ID", "Título", "Score BM25", "Rank BM25", "Texto"]].reset_index(drop=True), use_container_width=True)
        st.caption("A busca léxica é excelente para recuperar termos exatos (ex: CÓD-ECG-12D), mas falha em sinônimos.")
        
    with tab_sem:
        st.subheader("Resultados do Motor Semântico (Cosine Similarity)")
        df_sem = pd.DataFrame(results).sort_values(by="Score Semântico", ascending=False)
        st.dataframe(df_sem[["ID", "Título", "Score Semântico", "Rank Semântico", "Texto"]].reset_index(drop=True), use_container_width=True)
        st.caption("A busca semântica é excelente para capturar intenção e sinônimos, mas pode perder precisão em códigos e IDs exatos.")
        
    with tab_hyb:
        st.subheader("Resultados da Fusão (Reciprocal Rank Fusion)")
        if use_cross_encoder:
            st.success("Re-Ranking com Cross-Encoder ativado para os Top-3 candidatos!")
            cols_to_show = ["ID", "Título", "Score RRF", "Score Cross-Encoder", "Texto"]
        else:
            cols_to_show = ["ID", "Título", "Score RRF", "Texto"]
            
        df_hyb = pd.DataFrame(hybrid_results)
        st.dataframe(df_hyb[cols_to_show].reset_index(drop=True), use_container_width=True)
        
    with tab_comp:
        st.subheader("Matriz Comparativa de Desempenho")
        df_comp = pd.DataFrame(hybrid_results)
        
        # Adiciona coluna do Rank Final Híbrido para facilitar a comparação visual
        df_comp.insert(0, "Rank Híbrido", range(1, len(df_comp) + 1))
        
        cols_to_show_comp = ["Rank Híbrido", "ID", "Título", "Rank BM25", "Rank Semântico", "Score RRF"]
        if use_cross_encoder:
            cols_to_show_comp.append("Score Cross-Encoder")
            
        st.dataframe(df_comp[cols_to_show_comp].reset_index(drop=True), use_container_width=True)
        st.info("Observe a variação nos Ranks. A combinação RRF estabiliza o resultado trazendo os melhores de ambos os mundos para o topo.")
