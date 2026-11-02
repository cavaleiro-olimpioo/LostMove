# LostMove
 Sistema de detecção de analise de movimento e perca de movimento de um pêndulo


# Formas de uso

Adicione todos os videos que deseja testar, dentro da pasta data/

# Requeriments

Crie um venv com python3 -m venv .venv, ou no windows py -m venv .venv, logo após adicione a biblioteca numpy e opencv-python

# Escolher e analisar vídeos

1. Coloque os vídeos dentro da pasta `data/`. O programa também procura em subpastas.
2. Execute `.venv/bin/python main.py` no Linux ou `py main.py` no Windows.
3. No terminal, escolha o número do vídeo. `[NOVO]` indica que ainda não foi analisado; `[TESTADO]` indica que já há resultados na planilha.
4. Ao escolher um vídeo, o programa analisa todos os ciclos completos, mostra o vídeo e guarda os resultados em `results/pendulum_cycles.csv`.

Os resultados de vídeos diferentes ficam juntos na planilha. Se um vídeo já testado for escolhido novamente, somente os resultados antigos desse vídeo serão atualizados.

Formatos aceitos: MP4, AVI, MOV, MKV, M4V e WMV. Para analisar sem abrir as janelas, execute `.venv/bin/python main.py --sem-janelas` no Linux ou `py main.py --sem-janelas` no Windows.