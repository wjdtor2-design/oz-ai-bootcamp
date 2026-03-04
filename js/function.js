// 함수(function)

function add(num1, num2) {
    // 반환(return) = 호출부로 값을 되돌려주는 것
    return num1 + num2
}

// let myFunction = add;
// console.log(myFunction(10, 20));

// 1,2라는 인자값을 전달해서 add라는 함수를 "호출"
add(1, 2)

// add라는 함수 자체
add

// add 함수를 myFunction 변수에 할당
const myFunction = add;

// myFunction(=add)을 호출해서 반환값을 result 변수에 할당
const result = myFunction(10, 20);

// result 값을 출력
console.log(result);

// print라는 새로운 함수 정의
function print(message) {
    console.log(message);
    // return undefined
}

const printResult = print("hello");
console.log(printResult);