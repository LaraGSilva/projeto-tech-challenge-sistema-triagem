import sys


def main():
    print(" ==============  Iniciando a validação de ambiente para MLOps  ==============")


    version = sys.version_info
    print(f"Versão do pyhton: {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("Erro: o pipelone exige python 3.8 ou superior para machine learning.")
        sys.exit(1)

    print("Suceso: Ambiente validado e pronto para os próximos passos.")

if __name__ == "__main__":
    main()
