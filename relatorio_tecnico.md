# Relatório Técnico: HealthSearch — Motor de Busca Híbrido

**Instituição:** UNIPÊ — Centro Universitário de João Pessoa  
**Disciplina:** Tendências em Ciência da Computação (Recuperação de Informação / NLP)  
**Professor:** Me. Ricardo Roberto de Lima  
**Equipe:** Mateus Ieno Ramalho, Heitor de Oliveira Mamede, João Gabriel Barreto de Araújo Falcão
**Auxílio:** Gemini 3.1 pro

---

## 1. Arquitetura da Solução

A solução **HealthSearch** foi desenvolvida em Python e estruturada através do framework **Streamlit** para prover uma interface analítica interativa. O sistema endereça a problemática de "pontos cegos" na recuperação de informações médicas unindo duas abordagens distintas de busca através de um pipeline em 4 fases:

1. **Ingestão e Pré-processamento:** O sistema suporta tanto uma base local pré-carregada (6 diretrizes clínicas expandidas) quanto o upload de arquivos PDF (via `pypdf`). O texto passa por sanitização (remoção de caracteres especiais), normalização (minúsculas) e remoção de _stopwords_ em português usando a biblioteca `nltk`.
2. **Motor Léxico (BM25):** Utiliza-se a biblioteca `rank_bm25` (Okapi BM25). Este motor é fundamental para capturar correspondências exatas de siglas, códigos médicos (ex: _CÓD-ECG-12D_) e jargões precisos. A interface permite calibrar dinamicamente a saturação da frequência ($k_1$) e a normalização de comprimento ($b$).
3. **Motor Semântico Vetorial:** Emprega o modelo `all-MiniLM-L6-v2` da biblioteca `sentence-transformers` para converter textos em _embeddings_ densos. O ranqueamento é feito por Similaridade de Cosseno (utilizando `util.cos_sim`). Esta abordagem garante alta cobertura para sinônimos e intenções semânticas (ex: buscar por "infarto" e recuperar "isquemia miocárdica").
4. **Algoritmo Híbrido (RRF):** A convergência é feita pelo método _Reciprocal Rank Fusion_. As posições geradas pelo BM25 e pelo Motor Semântico são fundidas sob a fórmula:
   $$Score\_RRF = \alpha \times \left[\frac{1}{60 + Rank\_BM25}\right] + (1 - \alpha) \times \left[\frac{1}{60 + Rank\_Semantico}\right]$$
   O peso $\alpha$ pode ser ajustado para priorizar a precisão léxica ou a abrangência semântica.
5. **Re-Ranking (Desafio Bônus):** Através de um _Cross-Encoder_ (`ms-marco-MiniLM-L-6-v2`), os Top-3 resultados do pipeline híbrido são reavaliados por inferência conjunta, aprimorando a precisão final da relevância.

---

## 2. Comparação de Ranks (Gráfico/Tabela)

A eficiência do algoritmo híbrido (RRF) em suprir as deficiências individuais de cada motor pode ser observada na variação de ranks da **Matriz Comparativa** para a query _"infarto"_:

| ID do Documento |              Título              | Rank Léxico (BM25) | Rank Semântico | Rank Híbrido Final |
| :-------------- | :------------------------------: | :----------------: | :------------: | :----------------: |
| **Doc 2**       |  Guia de Farmacologia Cardíaca   |         5          |       1        |       **1**        |
| **Doc 6**       |    Procedimentos de UTI Geral    |         1          |       5        |       **2**        |
| **Doc 3**       | Diretriz de Hipertensão Arterial |         4          |       2        |       **3**        |

_Análise qualitativa:_ O BM25 costuma zerar o _score_ (jogando o rank para baixo) se a palavra exata não estiver presente. O motor Semântico consegue identificar o contexto cardíaco e posicionar o Doc 1 adequadamente. O RRF resgata a performance unificada: o documento forte em ambas as abordagens lidera, seguido pelos documentos semanticamente relevantes.

---
