"""flydnet.lab - 실험적 기능 (효과가 조건부이거나 증명되지 않은 것). 인터페이스가 바뀌거나 없어질 수 있음.
`import flydnet`에는 들어 있지 않음 - from flydnet import lab으로 따로 불러야 씀. 엔진(ConnectomeLayer)은 layer.attach로 붙는
부품의 자리만 가지고, 여기의 부품이 그 자리에 붙음

  from flydnet import lab
  g = lab.Growth(layer, allow=["PN>KC"], budget=5000)   # 추가 연결 (구조적 가소성) - layer.growth
  g.grow(1000, rule="random"), g.prune(0.2), g.extra_edges()
  lab.grow_contrastive(g, 1000, inputs=X, encoder=enc)   # 정답 없이 대조 학습으로 자리 고르기

결과·한계는 README "실험적 기능", 연구 예제는 저장소의 lab/ 폴더
"""
from .growth import Growth
from .contrastive import contrastive_loss, corrupt, grow_contrastive, info_nce, views

__all__ = ["Growth", "grow_contrastive", "contrastive_loss", "info_nce", "views", "corrupt"]
