// 배열(array)
let numbers = [10, "two", 30];

for (const [i, num] of numbers.entries()) {
    console.log(i + "번 index 값: " + num);
}