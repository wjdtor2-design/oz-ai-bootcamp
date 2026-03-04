// 두 변수 선언
let num1 = 10;
let num2 = 5;

// 연산자를 문자열로 정하기
let operator = '*';

// 연산자에 따라 계산 결과를 출력하기
let result;

if (operator === '+') {
    result = num1 + num2;
} else if (operator === '-') {
    result = num1 - num2;
} else if (operator === '*') {
    result = num1 * num2;
} else if (operator === '/') {
    result = num1 / num2;
} else {
    result = '알 수 없는 연산자입니다.';
}

console.log(`결과: ${result}`);