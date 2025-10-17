if __name__ == "__main__":
    # Ejecuta los módulos en orden relativo a este archivo
    import runpy
    import os

    base_dir = os.path.abspath(os.path.dirname(__file__)) + "/src"
    sequence = [
        "data_processing.py",
        "environment.py",
        "agent.py",
        "controller.py",
    ]

    for filename in sequence:
        path = os.path.join(base_dir, filename)
        if os.path.isfile(path):
            print(f"Ejecutando {filename} ...")
            runpy.run_path(path, run_name="__main__")
        else:
            print(f"No se encontró el archivo: {path}")