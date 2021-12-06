# TCC

Os scripts contidos são as funções para realizar a análise de dados de EMG e EEG 

Para a obtenção dos onsets e offsets é necessário chamar a função onset, passando uma lista de sinais de canais diferentes de um mesmo movimento juntamente com todos os paremtros a serem utilizados durante o algoritimo.
Para o caso da existencia de multiplos canais de movimento é possivel utilizar a funcao fix_onset, para agrupar os onsets e offsets dos canais em apenas um grupo de onsets e offsets, valido para todos os canais.
É possível realizar a visualização dos dados, juntamente com os onsets e offsets por meio da função plotar.
A obtenção do ERD dos sinais de EEG é feita manualmente utilizando as funções do código.
As demais funções são utilizadas pelo algoritmo de Onset, ou são funções extras para facilitar a criação das análises
