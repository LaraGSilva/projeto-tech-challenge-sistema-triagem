# Projeto Tech Challenge Fase 3

## Sistema de triagem para inferência automática de laudos médicos

Este sistema de triagem automática de laudos médicos tem como principal objetivo atuar na categorização de doenças com base em laudos médicos textuais. O projeto foi desenvolvido com o intuito de ter uma arquitetura robusta, de baixa latência, otimizada e com monitoramento contínuo.


1. A arquitetura escolhida para esse projeto foi a arquitetura online. A ideia central do projeto é categorizar os laudos escritos no momento da emissão de acordo com as especialidades existentes:
    - neoplasms
    - digestive system diseases
    - nervous system diseases
    - cardiovascular diseases
    - general pathological conditions

## Deploy (AWS ECR + ECS Fargate)

Como a inferência é **online** (resposta no momento da emissão do laudo), a API é
servida em container: a imagem vive no **Amazon ECR** e roda como serviço no
**Amazon ECS (Fargate)**, que expõe `GET /` (UI de teste), `GET /health`,
`GET /metrics`, `GET /docs` (Swagger) e `POST /predict`. O modelo
(`models/classifier.pkl`) vai **embutido na imagem** — o container não depende de
S3/EFS em runtime.

A raiz `/` serve um frontend simples ([src/app/index.html](src/app/index.html)):
campo de texto para o laudo e uma caixa com a classe prevista, a confiança e a
distribuição de probabilidades por especialidade.

### Esteira contínua (`.github/workflows/cd.yml`)

A cada push na `main`, o **CI** (`ci.yml`) roda lint + testes; ao passar, o **CD**
dispara automaticamente (`workflow_run`) e:

1. builda a imagem a partir do `Dockerfile` e faz push para o ECR
   (`triagem-app`), com as tags `:<sha>` e `:latest`;
2. injeta essa imagem em `.aws/task-definition.json`
   (`aws-actions/amazon-ecs-render-task-definition`);
3. registra a nova revisão da task definition e atualiza o service `triagem-app`
   no cluster `default` (`aws-actions/amazon-ecs-deploy-task-definition`),
   aguardando o rollout estabilizar.

Também dá para rodar sob demanda em **Actions → CD → Run workflow**.

### Configuração necessária

- **Secrets** (Settings → Secrets and variables → Actions): `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`. Como o ambiente é AWS Academy
  (credenciais temporárias), os três precisam ser **atualizados a cada sessão do
  lab** antes de rodar o deploy.
- **Variável opcional** `API_URL` (DNS do ALB, sem barra final) para habilitar o
  smoke test `GET /health` + `POST /predict` ao fim do deploy.
- `.aws/task-definition.json` reflete a task definition `default-triagem-app`. Se
  mudar CPU/memória, roles, log group ou subnets no console, reexporte com
  `aws ecs describe-task-definition --task-definition default-triagem-app --query taskDefinition`.
