[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
[Console]::InputEncoding = [System.Text.Encoding]::UTF8

$testApp = "..\cmake-build-debug\test\TinyTorch_test.exe"

Write-Host "[INFO] 正在获取所有测试用例..."
$allTestsRaw = & $testApp --gtest_list_tests

$currentTestSuite = ""
$allTests = @()
foreach ($line in $allTestsRaw) {
    if ($line -match '^(\w+)\.') {
        $currentTestSuite = $matches[1]
    } elseif ($line -match '^\s+(\S+)') {
        $subTest = $matches[1]
        $fullTestName = "$currentTestSuite.$subTest"
        $allTests += $fullTestName
    }
}

Write-Host "[DEBUG] 过滤后的测试用例列表："
$allTests | ForEach-Object { Write-Host $_ }

$targetTests = $allTests | Where-Object { $_ -like "TEST_cuda_benchmark.*" }

if ($targetTests.Count -eq 0) {
    Write-Host "[WARNING] 没有找到 TEST_cuda_benchmark 下的测试用例。"
} else {
    # 确保 output 文件夹存在
    $outputDir = "output"
    if (-not (Test-Path -Path $outputDir)) {
        New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
        Write-Host "[INFO] 已创建输出目录: $outputDir"
    }

    Write-Host "`n[INFO] 将要分析以下 CUDA 基准测试："
    $targetTests | ForEach-Object { Write-Host "  - $_" }

    foreach ($test in $targetTests) {
        # 生成输出文件名（包含 output 目录路径）
        $outputName = "$outputDir/report_$($test.Replace('.', '_'))"  # 例如：output/report_TEST_cuda_benchmark_fp32_gemm
        Write-Host "`n**************************************************"
        Write-Host "* 正在分析测试用例: $test"
        Write-Host "**************************************************"

        # 构建命令（添加 --trace=cuda）
        $command = "nsys profile --trace=cuda --output `"$outputName`" --force-overwrite true --stats true `"$testApp`" --gtest_filter=`"$test`""
        Invoke-Expression $command
    }
}
Write-Host "`n✅ 所有 CUDA 基准测试已完成。报告文件已生成在当前目录下。"