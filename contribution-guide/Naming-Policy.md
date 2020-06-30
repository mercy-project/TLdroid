## 네이밍 규칙
### 브랜치 네이밍 규칙
1. 일반 피쳐 : `NEW-XXXX` 형태로 붙입니다.
2. 핫픽스 : `HOTFIX-XXXX` 형태로 붙입니다.
3. version bump : `version-bump` 로 네이밍합니다.
4. 이 외의 다른 경우 브랜치 네이밍은 자유롭게 하되, underline(`_`) 문자는 사용하지 않습니다.

### 코드
1. static constants variable 은  `UPPER_CASE_UNDER_LINE_SEPERATE`  로
2. intent 나 fragment 로 넘기는 bundle 의 key 는  `hungarian_case`  로 합니다.
3. android component 를 지칭하는 변수(ex. TextView) userNameTextView 와 같이 full name 을 사용합니다.

### XML
1. `activity_something.xml` 아래의 컴포넌트  id는 `something_foo_bar` 형식으로 네이밍합니다.
2. [futurice guide](https://github.com/minsoopark/android-best-practices-kor)  에 나오는 방식을 기준으로 합니다.
    -   xml tag 의 self close 태그의 경우  `/>`  는 개행 처리
    -   xml tag 의 시작 태그의 경우  `>`  는 개행 처리
    -   string resource 의 경우 컨텍스트를  `.`  으로
~~~xml
<string name="error.message.network">Network error</string>
<string name="error.message.call">Call failed</string>
<string name="error.message.map">Map loading failed</string>
~~~
3. color resource 의 경우 단순히 hex 값을 쓰는게 아니라 color context 의 이름을 씁니다.
4. id 의 경우 추가로 생성하지 않습니다.
~~~xml
<FrameLayout
    android:id="@+id/container_player_control"
    android:layout_below="@+id/playing_album_layout"
    ... />
~~~

와 같은 경우 playing_album_layout id 가 여러번 추가되어서 debug 가 힘듭니다.
아래에서 위로 내려가면서 xml 을 짜면 @id 로 해결이 가능합니다.