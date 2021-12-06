import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import mne

mne.set_log_level("ERROR")


def media_movel(dados: np.ndarray, tamanho: int):
    tamanho = int(tamanho)
    quant = len(dados) + 1 - tamanho
    mm = np.zeros(quant)
    for i in range(quant):
        mm[i] = dados[i : i + tamanho].sum() / tamanho
    return mm


def retificar(dado):
    for i in range(len(dado)):
        dado[i] = abs(dado[i])
    return dado


def uniquetol(dado):
    tol = max(dado) * 10 ** -12
    value = np.sort(dado)
    index = np.argsort(dado)

    ia = [index[0]]
    ref = value[0]
    for i in range(1, len(value)):
        if abs(ref - value[i]) > tol:
            ia.append(index[i])
            ref = value[i]

    return ia


def rms(dado: np.ndarray) -> float:
    rms = np.sqrt((dado ** 2).sum() / len(dado))
    return rms


def baseline(dado: np.ndarray, tamanho: int, rank: int):
    retificado = retificar(dado)
    media = media_movel(retificado, tamanho)
    media = np.double(media)
    for i in range(len(media)):
        media[i] = round(media[i], 10)
    IA = uniquetol(media)

    # print(f'IA: {IA}')

    diferenca = len(dado) - len(media)

    if rank > len(IA):
        start = IA[-1]
    else:
        start = IA[rank] + diferenca / 2
    intervalo = []
    for i in range(
        int(start - np.floor(tamanho / 2)), int(start + np.floor(tamanho / 2) + 1)
    ):

        intervalo.append(i)

    segmento = [0] * len(intervalo)
    for i in range(len(intervalo)):
        segmento[i] = dado[intervalo[i]]
    mean = np.mean(segmento)
    std = np.std(segmento)

    return mean, std


def first_threshold(dados, media, desvio, base_desvio):
    temp = np.zeros(len(dados))
    for i in range(len(dados)):
        if abs(dados[i]) > (media + desvio * base_desvio):
            temp[i] = 1

    return temp


def plot_dado_onset(ax, dados, fs, valor, onset=[], offset=[]):

    print(dados)
    if len(dados) == 0:
        print("sem dados para plotagem")
        # erro, não existe dados para printar
    elif len(onset) == 0 or len(offset) == 0:
        tempo = [i / fs for i in range(len(dados))]
        ax.plot(tempo, dados)
        print("printando apenas dados")
        # plotar apenas os dados
    else:
        tempo = [i / fs for i in range(len(dados))]
        ax.plot(tempo, dados)

        algo = [
            [[valor, valor], [onset[j] / fs, offset[j] / fs]] for j in range(len(onset))
        ]
        for i in range(len(algo)):
            ax.plot(algo[i][1], algo[i][0], "k", label="linha")
            ax.plot(onset[i] / fs, valor, "go", label="onset")
            ax.plot(offset[i] / fs, valor, "ro", label="offset")
        pass


def second_threshold(dados: np.ndarray, quantidade: int):
    temp = dados[:].tolist()
    temp.insert(0, 0)
    temp.append(0)
    diferenca = np.diff(temp)
    posdif = np.zeros(len(diferenca)).astype("int64")
    negdif = np.zeros(len(diferenca)).astype("int64")
    for i in range(len(diferenca)):
        if diferenca[i] > 0:
            posdif[i] = 1
        elif diferenca[i] < 0:
            negdif[i] = 1
    pospoint = np.nonzero(posdif)[0]
    negpoint = np.nonzero(negdif)[0]
    passa = negpoint - pospoint
    for i in range(len(passa)):
        if passa[i] >= quantidade:
            passa[i] = 1
        else:
            passa[i] = 0
    onset = np.array([]).astype("int64")
    offset = np.array([]).astype("int64")
    for i in range(len(passa)):
        if passa[i]:
            onset = np.append(onset, pospoint[i])
            offset = np.append(offset, negpoint[i] - 1)
            # onset.append(pospoint[0][i])
            # offset.append(negpoint[0][i] - 1)
    return onset, offset


def third_threshold(onset, offset, offsize):
    if len(onset) == 0:
        return onset, offset
    onshift = onset[1:]
    offconc = offset[0:-1]
    ons = [onset[0]]
    offs = []
    for i in range(len(onshift)):
        if onshift[i] - offconc[i] > offsize:
            ons.append(onshift[i])
            offs.append(offconc[i])
    offs.append(offset[-1])
    return np.array(ons), np.array(offs)


def fourth_threshold(onset, offset, size):
    if len(onset) == 0:
        return onset, offset
    # ons = []
    # offs = []
    # for i in range(len(onset)):
    #    if offset[i] - onset[i] >= size:
    #        ons.append(onset[i])
    #        offs.append(offset[i])
    ons = onset[(offset - onset) >= size]
    offs = offset[(offset - onset) >= size]
    return ons, offs


def fifth_threshold(dado, onset, offset, quantidade):
    if len(onset) == 0:
        return onset, offset
    # ons = []
    # offs = []
    signalrms = np.zeros(len(onset))
    for i in range(len(onset)):
        signalrms[i] = rms(dado[onset[i] : offset[i]])
    mean = np.mean(signalrms)
    std = np.std(signalrms)
    ons = onset[
        (signalrms > (mean - std * quantidade))
        & (signalrms[i] < (mean + 5 * std * quantidade))
    ]
    offs = offset[
        (signalrms > (mean - std * quantidade))
        & (signalrms[i] < (mean + 5 * std * quantidade))
    ]
    # for i in range(len(onset)):
    #    if (signalrms[i] > (mean - std * quantidade)) and (
    #        signalrms[i] < (mean + 5 * std * quantidade)
    #    ):
    #        ons.append(onset[i])
    #        offs.append(offset[i])

    return ons, offs


def mais_um(onset, offset, menor=2, maior=6, fs=100):
    pontos_min = menor * fs
    pontos_max = maior * fs
    # for i in range(len(onset) - 1):
    #    if (offset[i] - onset[i] >= pontos_max) or (offset[i] - onset[i] <= pontos_min):
    #        onset[i] = -1
    #        offset[i] = -1
    ons = onset[(offset - onset < pontos_max) & (offset - onset > pontos_min)]
    offs = offset[(offset - onset < pontos_max) & (offset - onset > pontos_min)]

    return ons, offs


def mudar_freq(dado, razao):
    novo_tamanho = int((len(dado) // razao) // 1)
    resp = [0] * novo_tamanho
    for i in range(len(resp)):
        if (float(i * razao)).is_integer():
            resp[i] = dado[int(i * razao)]
        else:
            current = int(i * razao // 1)
            frac = i * razao % 1
            resp[i] = dado[current] * (1 - frac) + dado[current + 1] * frac
    return np.array(resp)


def pegar_dado(arquivo, keys):
    if type(keys) != list:
        return []
    dados = []
    raw = mne.io.read_raw_brainvision(arquivo)
    for key in keys:
        print(key)
        dados.append(raw.get_data(picks=key)[0])
    return dados


def filtrar(dado, frequencia_corte, frequencia, ordem=4, tipo="band"):
    if type(frequencia_corte) != list:
        print(f"frequencia precisa ser uma lista")
        return dado

    for i in frequencia_corte:
        if type(i) != int:
            print(f"frequencia precisa ser uma lista de inteiros")
            return dado

    if tipo == "band":
        if len(frequencia_corte) != 2:
            print(f"para filtro do tipo band, frequencia precisa de 2 valores")
            return dado
        else:
            filtro = []
            sos = signal.butter(
                ordem,
                [frequencia_corte[0], frequencia_corte[1]],
                "bandpass",
                fs=frequencia,
                output="sos",
            )
            for item in dado:
                valor = signal.sosfiltfilt(sos, item)
                filtro.append(valor)
            return filtro
    elif tipo == "low":
        if len(frequencia_corte) != 1:
            print(f"para filtro do tipo low, frequencia precisa de 1 valore")
            return dado
        else:
            filtro = []
            sos = signal.butter(
                ordem, frequencia_corte[0], "lowpass", fs=frequencia, output="sos"
            )
            for item in dado:
                valor = signal.sosfiltfilt(sos, item)
                filtro.append(valor)
            return filtro
    elif tipo == "high":
        if len(frequencia_corte) != 1:
            print(f"para filtro do tipo high, frequencia precisa de 1 valore")
            return dado
        else:
            filtro = []
            sos = signal.butter(
                ordem, frequencia_corte[0], "highpass", fs=frequencia, output="sos"
            )
            for item in dado:
                valor = signal.sosfiltfilt(sos, item)
                filtro.append(valor)
            return filtro
    else:
        print(f"algum outro erro detectado")
        return dado


def corrigir(dado, razao):
    resp = [mudar_freq(i, razao) for i in dado]
    return resp


def onset(dados, parametros):

    Lb = parametros[0]
    Kb = parametros[1]
    Nsd = parametros[2]
    Ton = parametros[3]
    Toff = parametros[4]
    Ts = parametros[5]
    Nnt = parametros[6]
    Tj = parametros[7]

    i = 1

    onsets, offsets = [], []

    for dado in dados:
        mean, std = baseline(dado, Lb, Kb)

        primeiro = first_threshold(
            dado, mean, std, Nsd
        )  # primeiro threshold, dados acima da media mais numero de desvios padrão

        ons, offs = second_threshold(
            primeiro, Ton
        )  # segundo threshold, sequencia de 1 com pelo menos {quantidade} de 1's seguidos

        ons, offs = third_threshold(
            ons, offs, Toff
        )  # terceiro threshold, junta sequencia de 1's distanciados de menos de {quantidade} de 0's seguidos

        ons, offs = fourth_threshold(
            ons, offs, Ts
        )  # quarto threshold, elimina sequencias de 1's que possuam menos de {quantidade} de 1's seguidos

        ons, offs = fifth_threshold(dado, ons, offs, Nnt)  # faz umas magicas ai

        ons, offs = third_threshold(ons, offs, Tj)
        onsets.append(ons)
        offsets.append(offs)
        print(f"fim do dado {i}")
        i += 1

    return onsets, offsets


def str2list(string):
    lista = string.replace("[", "").replace("]", "").replace("\n", " ")
    elemento = ""
    resposta = []
    for i in lista:
        if i != " ":
            elemento += i
        elif elemento != "":
            resposta.append(int(elemento))
            elemento = ""
    return resposta


def normalizar(dado, fs, t_start, t_fim):
    p_start = t_start * fs
    p_fim = t_fim * fs
    razao = dado[p_start:p_fim].mean()
    return dado / razao


def fix_onsets(dados, triggers, parametros):

    pontos = np.zeros(len(dados[0]))

    for i in range(len(pontos)):
        ins = np.count_nonzero(
            np.array(
                [
                    np.count_nonzero(triggers[0][0] >= i)
                    < np.count_nonzero(triggers[1][0] >= i),
                    np.count_nonzero(triggers[0][1] >= i)
                    < np.count_nonzero(triggers[1][1] >= i),
                    np.count_nonzero(triggers[0][1] >= i)
                    < np.count_nonzero(triggers[1][2] >= i),
                    np.count_nonzero(triggers[0][3] >= i)
                    < np.count_nonzero(triggers[1][3] >= i),
                    np.count_nonzero(triggers[0][4] >= i)
                    < np.count_nonzero(triggers[1][4] >= i),
                    np.count_nonzero(triggers[0][5] >= i)
                    < np.count_nonzero(triggers[1][5] >= i),
                ]
            )
        )

        if ins >= parametros[0]:
            pontos[i] = 1

    verdadeiros_onsets = np.array([]).astype("int64")
    verdadeiros_offsets = np.array([]).astype("int64")

    for i in range(len(pontos) - 1):
        if pontos[i + 1] < pontos[i]:
            verdadeiros_offsets = np.append(verdadeiros_offsets, i)
        if pontos[i + 1] > pontos[i]:
            verdadeiros_onsets = np.append(verdadeiros_onsets, i)

    if len(verdadeiros_onsets) != len(verdadeiros_offsets):
        verdadeiros_onsets = verdadeiros_onsets[:-1]

    verdadeiros_onsets, verdadeiros_offsets = third_threshold(
        verdadeiros_onsets, verdadeiros_offsets, parametros[1]
    )

    verdadeiros_onsets, verdadeiros_offsets = mais_um(
        verdadeiros_onsets, verdadeiros_offsets
    )

    return verdadeiros_onsets, verdadeiros_offsets


def plotar(
    dados, tempo, onsets=None, offsets=None, fs=None, altura=0.0001, juntar=False
):

    fig, eixos = plt.subplots(len(dados), 1, True, True)

    if juntar:
        algo = [
            [[altura, altura], [onsets[j] / fs, offsets[j] / fs]]
            for j in range(len(onsets))
        ]
        for ax, dado in zip(
            eixos,
            dados,
        ):
            ax.plot(tempo, dado)
            for i in range(len(algo)):
                ax.plot(algo[i][1], algo[i][0], "k", label="linha")
                ax.plot(onsets[i] / fs, altura, "go", label="onset")
                ax.plot(offsets[i] / fs, altura, "ro", label="offset")

    else:
        for ax, dado in zip(eixos, dados):
            ax.plot(tempo, dado)
    return fig


def retificar_dados(dados):
    resp = [retificar(i) for i in dados]
    return resp
