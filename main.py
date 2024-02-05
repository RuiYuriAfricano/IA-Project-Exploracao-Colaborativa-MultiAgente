import tkinter as tk
import random
import time

import pandas as pd
import random
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
# Importar os módulos necessários
from sklearn.neighbors import KNeighborsClassifier  # Importa o classificador KNN da scikit-learn
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
class Agente:
    def __init__(self, x, y, nomeAgente, modelo_escolhido, nomeModelo, cor):
        self.x = x
        self.y = y
        self.vivo = True
        self.nomeAgente = nomeAgente
        self.cor = cor
        self.encontrouBandeira = False
        self.tesouros = 0
        self.modelo = modelo_escolhido
        self.nomeModelo = nomeModelo
        self.lifeTime = 0
        self.ultimas_direcoes = []

    def mover(self, novo_x, novo_y, ambiente):
        # Adiciona a posição atual às últimas direções antes de mover
        self.ultimas_direcoes.append((self.x, self.y))
        # Mantém apenas as duas última posição
        self.ultimas_direcoes = self.ultimas_direcoes[-10:]

        if ambiente[self.x][self.y] == 'N':
          ambiente[self.x][self.y] = 'L'
        # Atualiza a posição do agente
        self.x = novo_x
        self.y = novo_y

    def atualizar_estado(self, base_conhecimento, bombas_conhecidas, tesouros_conhecidos, ambiente):

        # Atualiza o estado do agente (por exemplo, se encontrou uma bomba ou tesouro)
        if base_conhecimento[self.x][self.y] == 'B':
            if (self.tesouros - 1 < 0):
                self.vivo = False
                self.tesouros = 0
            else:
                self.tesouros -= 1
        elif base_conhecimento[self.x][self.y] == 'T':
            tesouros_conhecidos[self.x][self.y] = True
            self.tesouros += 1
        elif base_conhecimento[self.x][self.y] == 'F':
            self.encontrouBandeira = True

        # Atualiza a matriz de bombas conhecidas
        if base_conhecimento[self.x][self.y] == 'B':
            bombas_conhecidas[self.x][self.y] = True
        else:
            self.lifeTime += 1

        ambiente[self.x][self.y] = base_conhecimento[self.x][self.y]

class SimulationApp(tk.Tk):
    def __init__(self, tamanho, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title("Simulação de Exploração Multiagente")  # Define o título da janela
        self.tamanho = tamanho
        self.ambiente = [['N' for _ in range(tamanho)] for _ in range(tamanho)]
        self.baseDeConhecimento = self.inicializar_base_conhecimento(tamanho)
        self.total_tesouros_semeados = sum(linha.count('T') for linha in self.baseDeConhecimento)
        self.dadosTratados = self.tratarOsDados()
        self.agentes = self.criar_agentes_aleatorios(10)
        self.logsDaSimulacao = []
        self.initialize_ui()

        # Adiciona um atributo para armazenar o Label da abordagem selecionada
        self.label_abordagem_selecionada = tk.Label(self, text="Abordagem não selecionada")
        self.label_abordagem_selecionada.grid(row=5, column=self.tamanho + 1, sticky=tk.W)

        self.update_ambiente()
        #mecanismo de partilha de dados
        self.bombas_conhecidas = [[False for _ in range(tamanho)] for _ in range(tamanho)]
        self.tesouros_conhecidos = [[False for _ in range(tamanho)] for _ in range(tamanho)]

        #variáveis de controle
        self.simulacao_ativa = False
        self.simulacao_pausada = False

        self.inicio_simulacao = time.time()  # Marca o início da simulação
        self.total_bombas_semeadas = sum(linha.count('B') for linha in self.baseDeConhecimento)

        #tempo de vida
        self.max_iteracoes = 100  # Número máximo de iterações
        self.iteracao_atual = 0  # Contador de iteração atual

        self.abordagem_selecionada = None  # Adicionado para armazenar a abordagem selecionada

        # Atualiza as tarefas pendentes para obter as dimensões da janela
        self.update_idletasks()

        # Usa as dimensões da janela para calcular a posição central
        largura_janela = self.winfo_width()
        altura_janela = self.winfo_height()
        posicao_x = int(self.winfo_screenwidth() / 2 - largura_janela / 2)
        posicao_y = int(self.winfo_screenheight() / 2 - altura_janela / 2)

        # Configura a posição da janela
        self.geometry(f"+{posicao_x}+{posicao_y}")

    def tratarOsDados(self):
        df = pd.read_csv("random_data_combinations.csv")
        # Codificando as colunas categóricas
        label_encoder = LabelEncoder()
        # Normalizar os dados
        scaler = StandardScaler()
        for column in df.select_dtypes(include=['object']).columns:
            df[column] = label_encoder.fit_transform(df[column])
        return df
    def inicializar_base_conhecimento(self, tamanho):
        base = [['' for _ in range(tamanho)] for _ in range(tamanho)]
        total_celulas = tamanho * tamanho - 1  # Excluindo a célula com 'F'
        num_L_B = total_celulas // 3  # Distribuição igual de 'L' e 'B'

        # Lista de escolhas com quantidades iguais de 'L', 'B' e o restante 'T'
        escolhas = ['L'] * num_L_B + ['B'] * num_L_B + ['T'] * (total_celulas - 2 * num_L_B)
        random.shuffle(escolhas)  # Embaralhar as escolhas

        # Inserir exatamente um 'F' em uma posição aleatória
        f_posicao = (random.randint(0, tamanho - 1), random.randint(0, tamanho - 1))
        base[f_posicao[0]][f_posicao[1]] = 'F'

        # Preencher a base de conhecimento com as escolhas embaralhadas
        idx = 0
        for i in range(tamanho):
            for j in range(tamanho):
                if base[i][j] == '':  # Preencher apenas as células vazias
                    base[i][j] = escolhas[idx]
                    idx += 1

        return base
    def initialize_ui(self):
        self.cells = {}
        for i in range(self.tamanho):
            for j in range(self.tamanho):
                btn = tk.Button(self, text='', height=4, width=10)
                btn.grid(row=i, column=j)
                self.cells[(i, j)] = btn

                # Criando e configurando os botões com eventos de hover.
                self.btn_comecar = tk.Button(self, text="Começar/Continuar", bg="#4CAF50", cursor="hand2", command=self.iniciar_simulacao)
                self.btn_comecar.grid(row=0, column=self.tamanho + 1)
                self.btn_comecar.bind("<Enter>", lambda e: self.on_enter(self.btn_comecar, "#45a049"))
                self.btn_comecar.bind("<Leave>", lambda e: self.on_leave(self.btn_comecar, "#4CAF50"))

                self.btn_pausar = tk.Button(self, text="Pausar", bg="#007FFF",
                                            cursor="hand2", command=self.pausar_continuar_simulacao)  # Um azul sólido para o estado normal
                self.btn_pausar.grid(row=1, column=self.tamanho + 1)
                self.btn_pausar.bind("<Enter>", lambda e: self.on_enter(self.btn_pausar,
                                                                        "#1E90FF"))  # Um azul mais claro para o efeito hover
                self.btn_pausar.bind("<Leave>", lambda e: self.on_leave(self.btn_pausar,
                                                                        "#007FFF"))  # Voltar para a cor original quando o mouse sair

                self.btn_ver_resultado = tk.Button(self, text="Ver Resultado", state=tk.DISABLED, cursor="hand2", command=self.mostrarGrafico)
                self.btn_ver_resultado.grid(row=2, column=self.tamanho + 1)
                self.btn_ver_resultado.bind("<Enter>", lambda e: self.on_enter(self.btn_ver_resultado, "#bbbbbb"))
                self.btn_ver_resultado.bind("<Leave>", lambda e: self.on_leave(self.btn_ver_resultado, "SystemButtonFace"))

                self.btn_ver_msg_partilhadas = tk.Button(self, text="Ver Mensagens", cursor="hand2",
                                                   command=self.mostrar_mensagens)
                self.btn_ver_msg_partilhadas.grid(row=3, column=self.tamanho + 1)
                self.btn_ver_msg_partilhadas.bind("<Enter>", lambda e: self.on_enter(self.btn_ver_msg_partilhadas, "#bbbbbb"))
                self.btn_ver_msg_partilhadas.bind("<Leave>",
                                            lambda e: self.on_leave(self.btn_ver_msg_partilhadas, "SystemButtonFace"))

                self.iteracao_label = tk.Label(self, text="Iteração atual: 0", bg='black', fg='white')
                self.iteracao_label.grid(row=10, column=0, columnspan=self.tamanho, sticky=tk.EW)

                self.btn_ver_logs = tk.Button(self, text="Ver Logs", cursor="hand2",
                                                         command=self.mostrar_logs)
                self.btn_ver_logs.grid(row=4, column=self.tamanho + 1)
                self.btn_ver_logs.bind("<Enter>",
                                                  lambda e: self.on_enter(self.btn_ver_logs, "#bbbbbb"))
                self.btn_ver_logs.bind("<Leave>",
                                                  lambda e: self.on_leave(self.btn_ver_logs,
                                                                          "SystemButtonFace"))

    def on_enter(self, btn, color):
        btn['background'] = color

    def on_leave(self, btn, color):
        btn['background'] = color

    def criar_agentes_aleatorios(self, num_agentes):
        agentes = []

        nb = GaussianNB()
        knn = KNeighborsClassifier(n_neighbors=3)
        dtree_gini = DecisionTreeClassifier(criterion='gini')
        nn_model = MLPClassifier(max_iter=500)

        modelos_iniciais = [knn, nb, dtree_gini, nn_model]  # Substitua pelos seus modelos de aprendizado de máquina
        modelos = modelos_iniciais * (num_agentes // len(modelos_iniciais) + 1)
        random.shuffle(modelos)

        Xcolumns = self.dadosTratados.iloc[:, :-1]  # todas as colunas exceto a última
        y_columns = self.dadosTratados.iloc[:, -1]  # a última coluna

        # Dividindo a matriz em regiões
        num_regioes = int(num_agentes ** 0.5)
        regioes_utilizadas = set()

        for i in range(num_agentes):
            modelo_escolhido = modelos[i]
            nomeModelo = ""
            cor = ''

            if modelo_escolhido == knn:
                nomeModelo = "knn"
                cor = 'red'
            elif modelo_escolhido == nb:
                nomeModelo = "nb"
                cor = 'green'
            elif modelo_escolhido == dtree_gini:
                nomeModelo = "dtree"
                cor = 'orange'
            elif modelo_escolhido == nn_model:
                nomeModelo = "nn"
                cor = 'blue'

            # Garantindo a dispersão dos agentes na matriz
            while True:
                regiao_x = random.randint(0, num_regioes - 1)
                regiao_y = random.randint(0, num_regioes - 1)
                if (regiao_x, regiao_y) not in regioes_utilizadas or len(regioes_utilizadas) == num_regioes ** 2:
                    regioes_utilizadas.add((regiao_x, regiao_y))
                    break

            x = random.randint(regiao_x * self.tamanho // num_regioes, (regiao_x + 1) * self.tamanho // num_regioes - 1)
            y = random.randint(regiao_y * self.tamanho // num_regioes, (regiao_y + 1) * self.tamanho // num_regioes - 1)

            # Treinamento do modelo para cada agente
            X_train, X_test, y_train, y_test = train_test_split(Xcolumns, y_columns, test_size=0.3, random_state=42)
            modelo_escolhido.fit(X_train, y_train)

            agentes.append(Agente(x, y, f"Agent{i+1}", modelo_escolhido, nomeModelo, cor))

        return agentes

    def obter_posicoes_adjacentes(self, agente, ambiente):
        x, y = agente.x, agente.y
        tamanho = len(ambiente)
        posicoes_adjacentes = []

        # Direção Esquerda (E) ou alternativa
        if y > 0:
            posicoes_adjacentes.append({'valor': ambiente[x][y - 1], 'direcao': 'E', 'x': x, 'y': y - 1})
        elif x > 0:  # Se não pode ir para esquerda, tenta subir
            posicoes_adjacentes.append({'valor': ambiente[x - 1][y], 'direcao': 'E', 'x': x - 1, 'y': y})
        else:  # Se não pode subir, desce (garantido por estar na borda superior)
            posicoes_adjacentes.append({'valor': ambiente[x + 1][y], 'direcao': 'E', 'x': x + 1, 'y': y})

        # Direção Direita (D) ou alternativa
        if y < tamanho - 1:
            posicoes_adjacentes.append({'valor': ambiente[x][y + 1], 'direcao': 'D', 'x': x, 'y': y + 1})
        elif x < tamanho - 1:  # Se não pode ir para direita, tenta descer
            posicoes_adjacentes.append({'valor': ambiente[x + 1][y], 'direcao': 'D', 'x': x + 1, 'y': y})
        else:  # Se não pode descer, sobe (garantido por estar na borda inferior)
            posicoes_adjacentes.append({'valor': ambiente[x - 1][y], 'direcao': 'D', 'x': x - 1, 'y': y})

        # Direção Cima (C) ou alternativa
        if x > 0:
            posicoes_adjacentes.append({'valor': ambiente[x - 1][y], 'direcao': 'C', 'x': x - 1, 'y': y})
        elif y < tamanho - 1:  # Se não pode ir para cima, tenta direita
            posicoes_adjacentes.append({'valor': ambiente[x][y + 1], 'direcao': 'C', 'x': x, 'y': y + 1})
        else:  # Se não pode ir para direita, vai para esquerda (garantido por estar no canto direito)
            posicoes_adjacentes.append({'valor': ambiente[x][y - 1], 'direcao': 'C', 'x': x, 'y': y - 1})

        # Direção Baixo (BB) ou alternativa
        if x < tamanho - 1:
            posicoes_adjacentes.append({'valor': ambiente[x + 1][y], 'direcao': 'BB', 'x': x + 1, 'y': y})
        elif y > 0:  # Se não pode ir para baixo, tenta esquerda
            posicoes_adjacentes.append({'valor': ambiente[x][y - 1], 'direcao': 'BB', 'x': x, 'y': y - 1})
        else:  # Se não pode ir para esquerda, vai para direita (garantido por estar no canto esquerdo)
            posicoes_adjacentes.append({'valor': ambiente[x][y + 1], 'direcao': 'BB', 'x': x, 'y': y + 1})

        return posicoes_adjacentes

    # Função para exibir informações dos agentes
    def exibir_informacoes_agentes(self, agentes):
        i = 0
        logs = self.logsDaSimulacao
        logs.append(f"ITERACAO:{self.iteracao_atual}\n")
        for agente in agentes:
            i += 1
            logs.append(f"{agente.nomeAgente} = {agente.cor}<{agente.nomeModelo}>, Localização({agente.x}, {agente.y}): Vivo = {agente.vivo}, Encontrou Bandeira = {agente.encontrouBandeira}, Life QTD = {agente.tesouros}\n")
        self.logsDaSimulacao = logs
    def mapear_valor(self, entrada):
        # Dicionário para mapeamento dos valores ajustado
        mapeamento = {
            'E': 3, 'D': 2, 'C': 1, 'BB': 0,
            'B': 0, 'F': 1, 'L': 2, 'N': 3, 'T': 4
        }

        # Retorna o valor mapeado, se a entrada estiver no dicionário
        return mapeamento.get(entrada, "Entrada inválida")
    def update_ambiente(self):
        for i in range(self.tamanho):
            for j in range(self.tamanho):
                self.cells[(i, j)].configure(bg='white', fg='black', text=self.ambiente[i][j])

        for agente in self.agentes:
            if agente.vivo:
                valor_atual = self.ambiente[agente.x][agente.y]

                novo_valor = f"*{agente.nomeAgente}*{valor_atual}"

                self.cells[(agente.x, agente.y)].configure(bg=agente.cor, text=novo_valor)

    def mover_agentes(self):
        for agente in self.agentes:
            if agente.vivo:
                result = self.obter_posicoes_adjacentes(agente, self.ambiente)
                new_instance = [self.mapear_valor(posicao['valor']) for posicao in result]

                # Verificar se há tesouros conhecidos nas posições adjacentes
                # e que não estão nas últimas direções visitadas
                direcao_tesouro = next((item for item in result if self.tesouros_conhecidos[item['x']][item['y']] and (
                item['x'], item['y']) not in agente.ultimas_direcoes), None)

                if direcao_tesouro:
                    # Mover para o tesouro conhecido se não for uma das últimas posições visitadas
                    agente.mover(direcao_tesouro['x'], direcao_tesouro['y'], self.ambiente)
                    agente.atualizar_estado(self.baseDeConhecimento, self.bombas_conhecidas, self.tesouros_conhecidos,
                                            self.ambiente)
                else:
                    r = agente.modelo.predict([new_instance])[0]
                    mapeamento_direcoes = {3: 'E', 2: 'D', 1: 'C', 0: 'BB'}
                    letra = mapeamento_direcoes.get(r, '')

                    # Encontrar a direção correspondente ao valor previsto
                    # que também não esteja entre as últimas posições visitadas
                    direcao_movimento = next((item for item in result if item['direcao'] == letra and (
                    item['x'], item['y']) not in agente.ultimas_direcoes), None)

                    # Se todas as direções possíveis são inseguras ou foram visitadas recentemente, escolher aleatoriamente
                    if not direcao_movimento:
                        direcoes_possiveis = [item for item in result if
                                              (item['x'], item['y']) not in agente.ultimas_direcoes]
                        if direcoes_possiveis:
                            direcao_movimento = random.choice(direcoes_possiveis)
                        else:
                            # Se todas as direções foram visitadas recentemente, ignora a restrição uma vez para evitar deadlock
                            direcao_movimento = random.choice(result)

                    # Mover o agente para a nova posição escolhida
                    if direcao_movimento:
                        agente.mover(direcao_movimento['x'], direcao_movimento['y'], self.ambiente)
                        agente.atualizar_estado(self.baseDeConhecimento, self.bombas_conhecidas,
                                                self.tesouros_conhecidos, self.ambiente)

                self.update_ambiente()

    def mostrar_logs(self):
        logs_window = tk.Toplevel(self)
        logs_window.title("Logs da Simulação")

        # Criando um Frame para conter o Text e a Scrollbar
        container = tk.Frame(logs_window)
        container.pack(fill="both", expand=True)

        # Criando a Scrollbar dentro do container
        scrollbar = tk.Scrollbar(container)
        scrollbar.pack(side="right", fill="y")

        # Criando o Text dentro do container
        text_area = tk.Text(container, wrap="word", yscrollcommand=scrollbar.set)
        text_area.pack(side="left", fill="both", expand=True)

        # Configurando a Scrollbar para controlar o Text
        scrollbar.config(command=text_area.yview)

        # Inserindo os logs no text_area
        for log in self.logsDaSimulacao:
            text_area.insert('end', log + '\n')
        text_area.config(state='disabled')  # Desabilita a edição

    def mostrar_mensagens(self):
        logs_window = tk.Toplevel(self)
        logs_window.title("Mensagens Partilhadas")

        # Criando um Frame para conter o Text e a Scrollbar
        container = tk.Frame(logs_window)
        container.pack(fill="both", expand=True)

        # Criando a Scrollbar dentro do container
        scrollbar = tk.Scrollbar(container)
        scrollbar.pack(side="right", fill="y")

        # Criando o Text dentro do container
        text_area = tk.Text(container, wrap="word", yscrollcommand=scrollbar.set)
        text_area.pack(side="left", fill="both", expand=True)

        # Configurando a Scrollbar para controlar o Text
        scrollbar.config(command=text_area.yview)

        # Inserir informações sobre localizações conhecidas de bombas
        bombas_texto = "\n".join([f"Bomba em: ({i}, {j})" for i in range(self.tamanho) for j in range(self.tamanho) if
                                  self.bombas_conhecidas[i][j]])
        if bombas_texto:
            text_area.insert('end', "BOMBA!!! Mensagens Partilhadas entre os agentes:\n" + bombas_texto + '\n\n')
        else:
            text_area.insert('end', "Nenhuma bomba conhecida.\n\n")

        # Inserir informações sobre localizações conhecidas de tesouros
        tesouros_texto = "\n".join(
            [f"Tesouro em: ({i}, {j})" for i in range(self.tamanho) for j in range(self.tamanho) if
             self.tesouros_conhecidos[i][j]])
        if tesouros_texto:
            text_area.insert('end', "TESOURO!!! Mensagens Partilhadas entre os agentes:\n" + tesouros_texto + '\n\n')
        else:
            text_area.insert('end', "Nenhum tesouro conhecido.\n\n")

        text_area.config(state='disabled')  # Desabilita a edição

    def mostrarGrafico(self):
        # Mapeamento de modelos para cores
        cores_por_modelo = {
            'knn': 'red',
            'nn': 'blue',
            'nb': 'green',
            'dtree': 'orange'
        }

        # Agrupar agentes por modelo e calcular a média do lifeTime
        tempos_de_vida_por_modelo = {}
        for agente in self.agentes:
            if agente.nomeModelo not in tempos_de_vida_por_modelo:
                tempos_de_vida_por_modelo[agente.nomeModelo] = []
            tempos_de_vida_por_modelo[agente.nomeModelo].append(agente.lifeTime)

        media_tempo_de_vida = {nomeModelo: sum(tempos) / len(tempos)
                               for nomeModelo, tempos in tempos_de_vida_por_modelo.items()}

        # Encontrar o máximo tempo de vida médio
        max_tempo_vida = self.iteracao_atual

        # Calcular porcentagens
        percentagens = {modelo: (tempo / max_tempo_vida) * 100
                        for modelo, tempo in media_tempo_de_vida.items()}

        # Criar o gráfico de barras com porcentagens
        cores = [cores_por_modelo[modelo] for modelo in percentagens.keys()]
        plt.bar(percentagens.keys(), percentagens.values(), color=cores)
        plt.xlabel('Modelo IA')
        plt.ylabel('Tempo Médio de Vida (%)')
        plt.title('Tempo de Vida Médio por Modelo de IA (em %)')
        plt.show()

    def verificar_condicoes_de_sucesso(self):
        sucesso_A = self.verificar_sucesso_A()
        sucesso_B = self.verificar_sucesso_B()
        sucesso_C = self.verificar_sucesso_C()

        if self.abordagem_selecionada == 'A' and sucesso_A:
            return True
        elif self.abordagem_selecionada == 'B' and sucesso_B:
            return True
        elif self.abordagem_selecionada == 'C' and sucesso_C:
            return True
        elif self.abordagem_selecionada == 'MIX' and (sucesso_A or sucesso_B or sucesso_C):
            return True
        return False

    def total_tesouros_descobertos(self):
        tesouros_descobertos = 0
        for linha in self.ambiente:
            for celula in linha:
                if celula == 'T':
                    tesouros_descobertos += 1
        return tesouros_descobertos

    def ambiente_totalmente_explorado(self):
        for linha in self.ambiente:
            if 'N' in linha:  # Verifica se 'N' está presente em alguma linha
                return False  # Ambiente não foi totalmente explorado
        return True  # Não encontrou 'N' em nenhuma linha, ambiente totalmente explorado

    def pelo_menos_um_agente_vivo(self):
        for agente in self.agentes:  # Assumindo que self.agentes é a lista de agentes
            if agente.vivo:  # Se o agente está vivo
                return True  # Retorna True assim que encontrar o primeiro agente vivo
        return False  # Retorna False se nenhum agente vivo for encontrado

    def bandeira_encontrada_por_agente(self):
        for agente in self.agentes:  # Assumindo que self.agentes é a lista de agentes
            if agente.encontrouBandeira:  # Se o agente encontrou a bandeira
                return True  # Retorna True assim que encontrar o primeiro agente que encontrou a bandeira
        return False  # Retorna False se nenhum agente que encontrou a bandeira for encontrado

    def verificar_sucesso_A(self):
        # Verifica se o total de Tesouros descobertos é acima de 50% dos Tesouros semeados
        return self.total_tesouros_descobertos() > (self.total_tesouros_semeados / 2)

    def verificar_sucesso_B(self):
        # Verifica se o ambiente foi totalmente explorado e pelo menos um agente está vivo
        return self.ambiente_totalmente_explorado() and self.pelo_menos_um_agente_vivo()

    def verificar_sucesso_C(self):
        # Verifica se pelo menos um agente encontrou a Bandeira
        return self.bandeira_encontrada_por_agente()

    def finalizar_simulacao(self):
        # Calcula a duração da simulação
        duracao_simulacao = time.time() - self.inicio_simulacao

        # Atualiza a UI com as informações solicitadas
        info_str = f"Total de tesouros semeados: {self.total_tesouros_semeados}\n"
        info_str += f"Total de tesouros descobertos: {self.total_tesouros_descobertos()}\n"
        info_str += f"Quantidade de bombas: {self.total_bombas_semeadas}\n"
        info_str += f"Células livres: {sum(linha.count('L') for linha in self.baseDeConhecimento)}\n"
        info_str += f"Tempo da simulação: {duracao_simulacao:.2f} segundos"

        # Supondo que você tenha um label para exibir as informações
        self.label_informacoes_finais = tk.Label(self, text=info_str, justify=tk.LEFT)
        self.label_informacoes_finais.grid(row=6, column=self.tamanho + 1, sticky=tk.W)

        self.simulacao_ativa = False
        self.btn_ver_resultado['state'] = 'normal'
        self.btn_comecar['state'] = 'disabled'
        self.btn_pausar['state'] = 'disabled'
        # Adicione aqui qualquer lógica adicional de finalização, como mostrar resultados

    def processar_simulacao(self):
        if not self.simulacao_ativa or self.simulacao_pausada:
            #
            return

        # Coloque aqui a lógica para mover os agentes e atualizar o ambiente
        self.mover_agentes()
        self.update_ambiente()
        self.iteracao_atual += 1
        # OU para Tkinter, assumindo que você tenha um Label chamado self.iteracao_label
        self.iteracao_label.config(text=f"Iteração atual: {self.iteracao_atual}")
        self.exibir_informacoes_agentes(self.agentes)
        # Verifica se a simulação deve continuar
        if self.simulacao_ativa and not self.simulacao_pausada:
            # Se sim, agenda a próxima chamada de `processar_simulacao`
            self.after(800, self.processar_simulacao)  # Ajuste o tempo conforme necessário
        else:
            # Se a simulação estiver pausada ou não ativa, simplesmente retorna sem agendar a próxima chamada
            return

        # Condição para finalizar a simulação, se necessário
        # Por exemplo, se um certo critério for atingido (todos os agentes encontraram a bandeira, um limite de tempo foi alcançado, etc.)
        # Você pode chamar `self.finalizar_simulacao()` aqui
        if self.verificar_condicoes_de_sucesso():
            self.finalizar_simulacao()
            return  # Parar a simulação

    def pausar_continuar_simulacao(self):
        self.simulacao_pausada = not self.simulacao_pausada
        if not self.simulacao_pausada:
            # Continua a simulação
            self.btn_comecar['state'] = 'disabled'  # Desabilita o botão Começar/Continuar
            self.btn_pausar['state'] = 'normal'  # Habilita o botão Pausar
            self.processar_simulacao()  # Continua processando a simulação
        else:
            # Pausa a simulação
            self.btn_comecar['state'] = 'normal'  # Habilita o botão Começar/Continuar
            self.btn_pausar['state'] = 'disabled'  # Desabilita o botão Pausar

    def solicitar_abordagem_popup(self):
        self.popup = tk.Toplevel(self)
        self.popup.title("Seleção de Abordagem")
        self.popup.attributes("-topmost", True)  # Garante que o popup fique no topo
        largura_popup = 600
        altura_popup = 200

        # Calcula a posição x e y para posicionar a janela no centro da tela
        posicao_x = int(self.popup.winfo_screenwidth() / 2 - largura_popup / 2)
        posicao_y = int(self.popup.winfo_screenheight() / 2 - altura_popup / 2)

        # Configura a largura, altura e posição inicial do popup
        self.popup.geometry(f"{largura_popup}x{altura_popup}+{posicao_x}+{posicao_y}")
        self.popup.grab_set()  # Captura todos os eventos nesta janela

        tk.Label(self.popup, text="Selecione uma abordagem:").pack(pady=10)

        abordagens = [("Abordagem A", "A"), ("Abordagem B", "B"), ("Abordagem C", "C"), ("MIX", "MIX")]

        for text, abordagem in abordagens:
            tk.Button(self.popup, text=text, command=lambda ab=abordagem: self.definir_abordagem_e_iniciar(ab)).pack(
                fill=tk.X, padx=50, pady=5)

        self.popup.wait_window(self.popup)  # Espera a janela popup ser fechada

    def definir_abordagem_e_iniciar(self, abordagem):
        self.abordagem_selecionada = abordagem
        # Atualiza o Label com a abordagem selecionada
        self.label_abordagem_selecionada.config(text=f"Abordagem selecionada: {abordagem}")
        self.popup.destroy()  # Fecha a janela popup
        self.iniciar_simulacao()  # Inicia a simulação

    def iniciar_simulacao(self):
        if not self.simulacao_ativa:
            # Inicia a simulação pela primeira vez
            self.simulacao_ativa = True
            self.simulacao_pausada = False
            self.btn_comecar['state'] = 'disabled'  # Desabilita o botão Começar/Continuar
            self.btn_pausar['state'] = 'normal'  # Habilita o botão Pausar
            self.btn_ver_resultado['state'] = 'disabled'  # Desabilita o botão Ver Resultado
            self.processar_simulacao()
        elif self.simulacao_pausada:
            # Continua a simulação após uma pausa
            self.pausar_continuar_simulacao()

if __name__ == "__main__":
    ##suprime os warnings
    warnings.filterwarnings('ignore', category=UserWarning)
    app = SimulationApp(10)  # Tamanho do ambiente
    app.solicitar_abordagem_popup()  # Solicita a seleção de abordagem antes de iniciar
    #app.after(1000, app.iniciar_simulacao)  # Inicia a simulação após 1 segundo
    app.mainloop()
