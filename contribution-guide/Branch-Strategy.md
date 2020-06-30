## Branch 전략

 **Please check our [naming policy]() when you read this document.**

 ### 1.  Master
   - 배포 가능한 상태의 코드가 반영된 상태로 유지.
   - 포스 푸시 불가
   - 브랜치 삭제 불가
   - 마스터 브랜치 업데이트마다 태그(versionName.versionCode 포맷) 달기.

 ### 2.  Develop
   - 현재 개발 완료된 모든 피쳐들 반영된 상태로 유지
   - 포스 푸시 불가
   - 브랜치 삭제 불가
   - Default 브랜치로 설정

 ### 3.  Feature
   - 각 피쳐마다 현재의 develop 브랜치에서 브랜치 파서 생성함.
   - 작업 완료 후 풀 리퀘스트
   - 반드시 코드리뷰 approve 받은 후 merge
   - merge 할 때 squash merge 날려야 함 : 풀 리퀘스트 이름이 develop 브랜치의 커밋 메시지가 되도록.

 ### 4.  HotFix
   - 핫픽스 필요할 때 만드는 브랜치
   - 풀 리퀘스트 올리고 코드리뷰 받을 시간 없을경우 바로 merge.