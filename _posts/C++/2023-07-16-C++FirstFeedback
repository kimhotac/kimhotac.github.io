#C++ 1차 미팅 피드백
#### 객체, 멤버함수, 클래스의 정의. 파이썬과 차이점

|C++|python|설명|
|---|
|클래스|클래스|객체를 프로그램상에 인스턴스로 생성하는 설계도|
|인스턴스 변수|인스턴스 변수|인스턴스 생성시 전달받는 변수|
|멤버변수|멤버변수|클래스 내부에서 선언된 변수|
|멤버함수|메소드|클래스 내부에서 정의되어있는 함수, 현재 메서드로 일반화|
|인스턴스|인스턴스|클래스를 사용해 만든 실제 결과물|



멤버 함수의 정의에서 큰 차이를 보인다.
C++
> 클래스 내부에서 멤버 함수를 정의할 때 함수 이름 앞에 "ClassName::"를 붙여야 한다. 이는 클래스의 범위 내에서 함수를 정의한다는 것을 나타낸다.

파이썬
>클래스 내부에서 멤버 함수를 정의할 때 특별한 문법을 사용하지 않는다. 함수를 클래스 내부에 정의하면 된다.

```cpp
// 사람 클래스 정의
class Person {
public: //제한자
    // 멤버 변수 (데이터)
    std::string name;
    int age;
	
    // 멤버 함수
    void Person::introduce()
	{
        std::cout << "이름 : " << name << "나이 : " << age << std::endl;
    }
```

---

#####헷갈리는 부분 추가 정리
객체를 프로그램 상에 정의하기 위해서 여러 객체들의 특징을 모아 클래스를 만든다.
클래스를 통해 생성한 것이 인스턴스이다.

---

####signed char와 unsigned char를 예제를 이용하여 설명

signed char와 unsigned char을 사용하는 이유
- 명시적으로 표현하기 위해서
- char이 컴파일러에 따라 signed 또는 unsigned로 지정 가능하기 때문에 정수로 사용할 경우

|형태|표현범위|
|---|---|
|char|-128~127|
|siggned char|-128~127|
|unsigned char|0~255|

```cpp
#include <iostream>

using namespace std;

int main()
{
    unsigned char u_c = 65; //A의 ascii코드
    signed char s_c = 65; //A의 ascii코드
    int a = 65; //65 정수값
    cout << "u_c : " << u_c <<", s_c : "<< s_c;
    cout << endl << "char(int) : " << char(a);
    return 0;
}
```

output
>u_c : A, s_c : A
char(int) : A

위처럼 코드에서 cout은 char을 문자열로만 출력한다. 정수형으로 출력하려면 형 변환이 필요하다.
