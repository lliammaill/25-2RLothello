# 25-2RLothello
25년 2학기 Reinforcement Learning 과제

본 프로젝트는 2025년 2학기 강화 학습(Reinforcement Learning) 과제로, 고전 게임 오델로(Othello/Reversi) 환경에서 Q-Learning과 Deep Q-Network (DQN) 두 가지 핵심 RL 알고리즘을 구현하고 성능을 비교합니다.

1. 목표
테이블 기반(Table-Based) 학습 방식인 Q-Learning 구현.
딥러닝 기반(Deep Learning-Based) 학습 방식인 DQN 구현.
두 학습 모델을 상호 대결시켜 성능, 학습 효율성, 일반화 능력을 정량적으로 평가합니다.

2. 파일별 기능 소개
1) othello_env_agent : 
  a. 오델로 환경 (OthelloEnv) 정의 (상태, 보상, 전이 규칙 포함).
  b. Q-Learning Agent (QLearningAgent) 클래스 정의.
  c. DQN Agent (DQNAgent) 클래스 정의.
2) train_q_learning
  QLearningAgent를 초기화하고, 에피소드(예: 200,000회) 동안 오델로 환경에서 학습을 진행합니다. 학습된 Q-Table을 저장합니다.
  결과물 : q_table_othello.pkl
3) train_dqn : DQN 학습 모델
  DQNAgent (CNN 기반 모델)를 초기화, 상대적으로 적은 에피소드(예: 50,000회) 동안 학습을 진행합니다. 학습된 모델 가중치를 저장합니다.
  결과물 : dqn_othello_model.pth
4) evaluate_agents
  미리 학습된 Q-Learning Agent와 DQN Agent를 불러와 지정된 횟수만큼 대결을 진행하고 승률 및 통계 데이터를 출력합니다. 성능 비교 및 분석의 핵심 파일입니다

3. 실행 환경
1) Python(3.8 이상 권장)
2) 설치 필요 라이브러리
  a. pip install numpy
  b. pip install pandas  # 데이터 로깅 및 분석용 (선택 사항)
  c. pip install torch   # 또는 tensorflow (DQN 구현 시 선택)

4. 실행 방법
- DQN 산출물(dqn_othello_model.pth) Evaluate_Agents 파일을 한 폴더에 다운로드
- train_q_learning 파일을 다운로드(Q-learning 산출물 파일 크기가 1GB이상으로, 업로드가 불가함)하여 실행
- Evaluate_agents 파일을 실행시켜 결과를 확인한다. 

결과 예시. 
[10/100 게임 완료] DQN 승: 10, QL 승: 0, 무승부: 0
[20/100 게임 완료] DQN 승: 20, QL 승: 0, 무승부: 0
[30/100 게임 완료] DQN 승: 30, QL 승: 0, 무승부: 0
[40/100 게임 완료] DQN 승: 40, QL 승: 0, 무승부: 0
[50/100 게임 완료] DQN 승: 50, QL 승: 0, 무승부: 0
[60/100 게임 완료] DQN 승: 50, QL 승: 10, 무승부: 0
[70/100 게임 완료] DQN 승: 50, QL 승: 20, 무승부: 0
[80/100 게임 완료] DQN 승: 50, QL 승: 30, 무승부: 0
[90/100 게임 완료] DQN 승: 50, QL 승: 40, 무승부: 0
[100/100 게임 완료] DQN 승: 50, QL 승: 50, 무승부: 0
