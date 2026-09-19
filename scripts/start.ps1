[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Assert-Command {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Name
    )

    if (-not (
        Get-Command `
            $Name `
            -ErrorAction SilentlyContinue
    )) {
        throw (
            "Required command was not found: " +
            $Name
        )
    }
}


function Assert-LastExitCode {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if ($LASTEXITCODE -ne 0) {
        throw $Message
    }
}


function Remove-DvcPullContainer {
    try {
        docker compose rm `
            -s `
            -f `
            dvc-pull `
            *> $null
    }
    catch {
        # Best-effort cleanup only.
    }
}


$repoRoot = (
    Resolve-Path (
        Join-Path `
            $PSScriptRoot `
            ".."
    )
).Path

$previousDagshubToken = (
    $env:DAGSHUB_TOKEN
)

$startupSucceeded = $false

Push-Location $repoRoot

try {
    Write-Host "=== OLIST MLOPS STARTUP ==="
    Write-Host "Repository: $repoRoot"

    Write-Host ""
    Write-Host "=== CHECK DOCKER ==="

    Assert-Command `
        -Name "docker"

    docker info *> $null

    Assert-LastExitCode `
        -Message (
            "Docker is installed but the " +
            "Docker daemon is not available."
        )

    docker compose version

    Assert-LastExitCode `
        -Message (
            "Docker Compose is not available."
        )

    Write-Host "Docker: READY"

    Write-Host ""
    Write-Host "=== LOCAL ENVIRONMENT ==="

    $envPath = Join-Path `
        $repoRoot `
        ".env"

    if (-not (
        Test-Path `
            -LiteralPath $envPath
    )) {
        $password = (
            [guid]::NewGuid().ToString(
                "N"
            ) +
            [guid]::NewGuid().ToString(
                "N"
            ).Substring(
                0,
                16
            )
        )

        $envContent = @(
            "MLFLOW_DB_NAME=mlflow"
            "MLFLOW_DB_USER=mlflow_user"
            (
                "MLFLOW_DB_PASSWORD=" +
                $password
            )
            ""
        ) -join "`n"

        $utf8NoBom = New-Object `
            System.Text.UTF8Encoding(
                $false
            )

        [System.IO.File]::WriteAllText(
            $envPath,
            $envContent,
            $utf8NoBom
        )

        Write-Host (
            ".env created with a generated " +
            "local database password."
        )
    }
    else {
        Write-Host (
            "Existing .env will be used."
        )
    }

    $requiredEnvNames = @(
        "MLFLOW_DB_NAME"
        "MLFLOW_DB_USER"
        "MLFLOW_DB_PASSWORD"
    )

    $envText = Get-Content `
        -LiteralPath $envPath `
        -Raw

    foreach (
        $requiredName
        in $requiredEnvNames
    ) {
        if (
            $envText -notmatch (
                "(?m)^\s*" +
                [regex]::Escape(
                    $requiredName
                ) +
                "\s*=\s*\S+"
            )
        ) {
            throw (
                "Missing or empty variable " +
                "in .env: " +
                $requiredName
            )
        }
    }

    Write-Host "Local environment: READY"

    Write-Host ""
    Write-Host "=== DAGSHUB CREDENTIAL ==="

    $dagshubToken = (
        $env:DAGSHUB_TOKEN
    )

    if (
        [string]::IsNullOrWhiteSpace(
            $dagshubToken
        )
    ) {
        $secureToken = Read-Host `
            "Enter DagsHub token" `
            -AsSecureString

        $bstr = (
            [Runtime.InteropServices.Marshal]::
            SecureStringToBSTR(
                $secureToken
            )
        )

        try {
            $dagshubToken = (
                [Runtime.InteropServices.Marshal]::
                PtrToStringBSTR(
                    $bstr
                )
            )
        }
        finally {
            [Runtime.InteropServices.Marshal]::
            ZeroFreeBSTR(
                $bstr
            )
        }
    }

    if (
        [string]::IsNullOrWhiteSpace(
            $dagshubToken
        )
    ) {
        throw (
            "DagsHub token cannot be empty."
        )
    }

    $env:DAGSHUB_TOKEN = (
        $dagshubToken
    )

    Write-Host (
        "DagsHub credential loaded securely."
    )

    Write-Host ""
    Write-Host "=== VALIDATE COMPOSE ==="

    docker compose config --quiet

    Assert-LastExitCode `
        -Message (
            "Docker Compose configuration " +
            "is invalid."
        )

    Write-Host "Compose configuration: VALID"

    Write-Host ""
    Write-Host "=== START FULL STACK ==="

    docker compose up `
        --build `
        --detach `
        --remove-orphans

    Assert-LastExitCode `
        -Message (
            "Docker Compose startup failed."
        )

    Write-Host ""
    Write-Host "=== WAIT FOR API ==="

    $healthUri = (
        "http://127.0.0.1:8000/health"
    )

    $modelInfoUri = (
        "http://127.0.0.1:8000/model-info"
    )

    $deadline = (
        Get-Date
    ).AddMinutes(
        5
    )

    $health = $null

    while (
        (Get-Date) -lt $deadline
    ) {
        try {
            $health = Invoke-RestMethod `
                -Uri $healthUri `
                -Method Get `
                -TimeoutSec 5

            if (
                $health.status -eq "ok"
            ) {
                break
            }
        }
        catch {
            $health = $null
        }

        Start-Sleep `
            -Seconds 3
    }

    if (
        $null -eq $health -or
        $health.status -ne "ok"
    ) {
        Write-Host ""
        Write-Host "=== COMPOSE STATUS ==="

        docker compose ps

        Write-Host ""
        Write-Host "=== RECENT SERVICE LOGS ==="

        docker compose logs `
            --no-color `
            --tail 100 `
            dvc-pull `
            mlflow `
            model-bootstrap `
            api

        throw (
            "API did not become healthy " +
            "within 5 minutes."
        )
    }

    Write-Host "API health: OK"

    Write-Host ""
    Write-Host "=== MODEL INFO ==="

    $modelInfo = Invoke-RestMethod `
        -Uri $modelInfoUri `
        -Method Get `
        -TimeoutSec 10

    Write-Host (
        "Model name    : " +
        $modelInfo.registered_model_name
    )

    Write-Host (
        "Model alias   : " +
        $modelInfo.alias
    )

    Write-Host (
        "Model version : " +
        $modelInfo.version
    )

    Write-Host (
        "Model type    : " +
        $modelInfo.model_type
    )

    $startupSucceeded = $true

    Write-Host ""
    Write-Host "=== CLEAN DVC CREDENTIAL CONTAINER ==="

    Remove-DvcPullContainer

    Write-Host (
        "DVC pull container removed."
    )

    Write-Host ""
    Write-Host "=== SYSTEM READY ==="

    Write-Host (
        "API    : http://localhost:8000"
    )

    Write-Host (
        "Docs   : http://localhost:8000/docs"
    )

    Write-Host (
        "MLflow : http://localhost:5000"
    )

    Write-Host ""
    Write-Host (
        "OLIST MLOPS STARTUP: PASSED"
    )
}
catch {
    Write-Host ""
    Write-Host "STARTUP FAILED"

    Remove-DvcPullContainer

    throw
}
finally {
    if (
        $null -eq $previousDagshubToken
    ) {
        Remove-Item `
            Env:DAGSHUB_TOKEN `
            -ErrorAction SilentlyContinue
    }
    else {
        $env:DAGSHUB_TOKEN = (
            $previousDagshubToken
        )
    }

    Pop-Location
}
