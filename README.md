# Projeto Tech Challenge Fase 3

## Sistema de triagem para inferência automática de laudos médicos

Este sistema de triagem automática de laudos médicos tem como principal objetivo atuar na categorização de doenças com base em laudos médicos textuais. O projeto foi desenvolvido com o intuito de ter uma arquitetura robusta, de baixa latência, otimizada e com monitoramento contínuo.


1. A arquitetura escolhida para esse projeto foi a arquitetura bacth. A ideia central do projeto é categorizar os laudos escritos durante um dia de atendimento na clinica de acordo com as especialidades existentes no ambulatório:
    - neoplasms
    - digestive system diseases
    - nervous system diseases
    - cardiovascular diseases
    - general pathological conditions
A escolha da arquitetura batch deve-se ao fato de que ao realizar um deploy bacth a aplicação do modelo em produção é realizada em cima de um conjunto de dados previamente armazenado, processado e executado em forma programatica.Caracteristica que atende a demanda de negócio disponibilizada no projeto.

