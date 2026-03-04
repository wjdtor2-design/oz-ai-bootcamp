// 논리 연산자 : 여러 조건의 조합(&&, ||)
// &&(AND) = 둘 다 만족해야지만 ture
// ||(OR) = 둘 중 하나만 true면 true

let isAdult = true;
let hasTicket = false;

let canPass = isAdult || hasTicket;
console.log(canPass);