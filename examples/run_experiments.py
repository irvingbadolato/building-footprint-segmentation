import os
import nbformat
from datetime import datetime
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError

def run_experiment(nb_path, timeout=4*3600):
    """
    Executa todas as células de um Jupyter Notebook e salva o arquivo de volta no disco,
    mesmo que ocorra uma falha durante a execução.
    
    :param caminho_notebook: Str. Caminho para o arquivo .ipynb
    :param timeout_por_celula: Int. Tempo máximo (em segundos) esperado para cada célula.
    """
    if not os.path.exists(nb_path):
        print(f"{datetime.now()} - Erro: O arquivo '{nb_path}' não foi encontrado.", flush=True)
        return

    print(f"{datetime.now()} - Lendo o notebook: {nb_path}...", flush=True)
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    # 2. Configura o executor
    # allow_errors=True garante que ele continue processando ou salve o estado atual em caso de falha
    ep = ExecutePreprocessor(timeout=timeout, kernel_name='python3', allow_errors=True)

    print("Iniciando a execução das células...", flush=True)
    try:
        # Executa o notebook no diretório onde ele está localizado
        base_path = os.path.dirname(nb_path) or '.'
        ep.preprocess(nb, {'metadata': {'path': base_path}})
        print(f"{datetime.now()} - Execução concluída com sucesso!", flush=True)
        
    except CellExecutionError as e:
        # Nota: Como 'allow_errors=True' está ativo, este bloco geralmente só será ativado
        # se houver um erro catastrófico no kernel (e não um erro comum de código Python).
        print(f"{datetime.now()} - Ocorreu um erro crítico durante a execução do kernel: {e}", flush=True)
        
    finally:
        # 3. Salva o arquivo de qualquer forma (com os outputs gerados até o momento do erro)
        print(f"{datetime.now()} - Salvando as alterações em: {nb_path}...", flush=True)
        with open(nb_path, 'w', encoding='utf-8') as f:
            nbformat.write(nb, f)
        print("Arquivo atualizado e salvo!", flush=True)

# --- Exemplo de Uso ---
if __name__ == "__main__":
    run_experiment("Experiment_0_RefineNet_resnet50_massachusetts.ipynb")
    run_experiment("Experiment_1_RefineNet_resnet50_rio.ipynb")
    run_experiment("Experiment_2_UNet_fullytrained_rio.ipynb")
    run_experiment("Experiment_3_ResUNet_fullytrained_rio.ipynb")
    run_experiment("Experiment_4_RefineNet_fullytrained_rio.ipynb")
    run_experiment("Experiment_5_UNet_resnet50_rio.ipynb")
    run_experiment("Experiment_6_ResUNet_resnet50_rio.ipynb")
    run_experiment("Experiment_7_RefineNet_resnet50_rio_uncured.ipynb")
    run_experiment("Experiment_0_RefineNet_resnet50_massachusetts.ipynb")
    run_experiment("Experiment_1_RefineNet_resnet50_rio.ipynb")
    run_experiment("Experiment_2_UNet_fullytrained_rio.ipynb")
    run_experiment("Experiment_3_ResUNet_fullytrained_rio.ipynb")
    run_experiment("Experiment_4_RefineNet_fullytrained_rio.ipynb")
    run_experiment("Experiment_5_UNet_resnet50_rio.ipynb")
    run_experiment("Experiment_6_ResUNet_resnet50_rio.ipynb")
    run_experiment("Experiment_7_RefineNet_resnet50_rio_uncured.ipynb")
    run_experiment("Experiment_0_RefineNet_resnet50_massachusetts.ipynb")
    run_experiment("Experiment_1_RefineNet_resnet50_rio.ipynb")
    run_experiment("Experiment_2_UNet_fullytrained_rio.ipynb")
    run_experiment("Experiment_3_ResUNet_fullytrained_rio.ipynb")
    run_experiment("Experiment_4_RefineNet_fullytrained_rio.ipynb")
    run_experiment("Experiment_5_UNet_resnet50_rio.ipynb")
    run_experiment("Experiment_6_ResUNet_resnet50_rio.ipynb")
    run_experiment("Experiment_7_RefineNet_resnet50_rio_uncured.ipynb")
    
