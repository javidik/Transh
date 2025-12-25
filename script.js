// Mortgage Tranche Calculator
class MortgageCalculator {
    constructor() {
        this.paymentChart = null;
        this.structureChart = null;
        this.balanceChart = null;
        this.trancheCounter = 0;
        this.initializeEventListeners();
        this.addTranche(); // Add initial tranche
    }

    initializeEventListeners() {
        document.getElementById('addTranche').addEventListener('click', () => this.addTranche());
        document.getElementById('mortgageForm').addEventListener('submit', (e) => this.handleSubmit(e));
        document.getElementById('resetCalculator').addEventListener('click', () => this.resetCalculator());
    }

    addTranche() {
        const tranchesList = document.getElementById('tranchesList');
        this.trancheCounter++;
        
        const trancheId = `tranche_${this.trancheCounter}`;
        const trancheHtml = `
            <div class="tranche-item" data-tranche-id="${trancheId}">
                <div class="tranche-item-header">
                    <div class="tranche-item-title">Транш #${this.trancheCounter}</div>
                    <button type="button" class="remove-tranche" onclick="calculator.removeTranche('${trancheId}')">×</button>
                </div>
                <div class="tranche-fields">
                    <div class="form-group">
                        <label for="trancheAmount_${this.trancheCounter}">Сумма транша (₽):</label>
                        <input type="number" id="trancheAmount_${this.trancheCounter}" name="trancheAmount" min="0" step="1000" value="${this.trancheCounter === 1 ? '3000000' : '1000000'}" required>
                    </div>
                    <div class="form-group">
                        <label for="trancheRate_${this.trancheCounter}">Процентная ставка (%):</label>
                        <input type="number" id="trancheRate_${this.trancheCounter}" name="trancheRate" min="0" max="100" step="0.01" value="${this.trancheCounter === 1 ? '12' : '13'}" required>
                    </div>
                    <div class="form-group">
                        <label for="trancheStartDate_${this.trancheCounter}">Дата выдачи:</label>
                        <input type="date" id="trancheStartDate_${this.trancheCounter}" name="trancheStartDate" value="${this.formatDate(new Date())}" required>
                    </div>
                </div>
            </div>
        `;
        
        tranchesList.insertAdjacentHTML('beforeend', trancheHtml);
    }

    removeTranche(trancheId) {
        const trancheElement = document.querySelector(`[data-tranche-id="${trancheId}"]`);
        if (trancheElement) {
            trancheElement.remove();
        }
    }

    formatDate(date) {
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        return `${year}-${month}-${day}`;
    }

    handleSubmit(e) {
        e.preventDefault();
        
        // Get form data
        const baseRate = parseFloat(document.getElementById('baseRate').value);
        const totalAmount = parseFloat(document.getElementById('totalAmount').value);
        const paymentType = document.getElementById('paymentType').value;
        const loanTerm = parseInt(document.getElementById('loanTerm').value);
        
        // Get tranches data
        const trancheElements = document.querySelectorAll('.tranche-item');
        const tranches = [];
        
        trancheElements.forEach((element, index) => {
            const trancheAmount = parseFloat(document.getElementById(`trancheAmount_${index + 1}`).value);
            const trancheRate = parseFloat(document.getElementById(`trancheRate_${index + 1}`).value);
            const trancheStartDate = document.getElementById(`trancheStartDate_${index + 1}`).value;
            
            if (trancheAmount && trancheRate && trancheStartDate) {
                tranches.push({
                    amount: trancheAmount,
                    rate: trancheRate,
                    startDate: new Date(trancheStartDate)
                });
            }
        });
        
        if (tranches.length === 0) {
            alert('Добавьте хотя бы один транш');
            return;
        }
        
        // Calculate mortgage
        const results = this.calculateMortgage(baseRate, totalAmount, paymentType, loanTerm, tranches);
        
        // Display results
        this.displayResults(results);
        
        // Show dashboard
        document.getElementById('dashboard').style.display = 'block';
        
        // Scroll to dashboard
        document.getElementById('dashboard').scrollIntoView({ behavior: 'smooth' });
    }

    calculateMortgage(baseRate, totalAmount, paymentType, loanTerm, tranches) {
        // Sort tranches by date
        tranches.sort((a, b) => a.startDate - b.startDate);
        
        // Calculate monthly interest rates
        const monthlyBaseRate = baseRate / 100 / 12;
        const monthlyTrancheRates = tranches.map(tranche => tranche.rate / 100 / 12);
        
        // Total months
        const totalMonths = loanTerm * 12;
        
        // Initialize payment schedule
        let paymentSchedule = [];
        let totalDebt = 0;
        
        // Calculate initial debt from all tranches
        tranches.forEach(tranche => {
            totalDebt += tranche.amount;
        });
        
        // Start date is the earliest tranche date
        const startDate = new Date(Math.min(...tranches.map(t => t.startDate)));
        
        // For simplicity in this example, we'll implement a basic model
        // where all tranches are considered together with weighted average rate
        let currentDebt = totalDebt;
        let currentDate = new Date(startDate);
        let monthCounter = 0;
        
        // Calculate weighted average rate
        let weightedRate = 0;
        let totalAmountWeight = 0;
        
        tranches.forEach(tranche => {
            weightedRate += tranche.amount * (tranche.rate / 100);
            totalAmountWeight += tranche.amount;
        });
        
        weightedRate = weightedRate / totalAmountWeight;
        const monthlyWeightedRate = weightedRate / 12;
        
        // Calculate monthly payment based on payment type
        let monthlyPayment = 0;
        
        if (paymentType === 'annuity') {
            // Annuity payment formula
            const n = totalMonths;
            const r = monthlyWeightedRate;
            
            if (r > 0) {
                monthlyPayment = (currentDebt * r * Math.pow(1 + r, n)) / (Math.pow(1 + r, n) - 1);
            } else {
                monthlyPayment = currentDebt / n;
            }
        }
        
        // Generate payment schedule
        for (let i = 0; i < totalMonths && currentDebt > 0; i++) {
            const monthInterest = currentDebt * monthlyWeightedRate;
            
            let principalPayment, interestPayment, totalPayment;
            
            if (paymentType === 'annuity') {
                interestPayment = monthInterest;
                principalPayment = Math.min(monthlyPayment - interestPayment, currentDebt);
                totalPayment = principalPayment + interestPayment;
            } else { // differentiated
                principalPayment = currentDebt / (totalMonths - i);
                interestPayment = monthInterest;
                totalPayment = principalPayment + interestPayment;
            }
            
            // Update debt
            currentDebt -= principalPayment;
            
            // Ensure debt doesn't go negative
            if (currentDebt < 0) {
                principalPayment += currentDebt;
                totalPayment = principalPayment + interestPayment;
                currentDebt = 0;
            }
            
            // Calculate the date for this payment
            const paymentDate = new Date(currentDate);
            paymentDate.setMonth(paymentDate.getMonth() + i + 1);
            
            paymentSchedule.push({
                month: i + 1,
                date: paymentDate.toLocaleDateString('ru-RU'),
                balance: currentDebt,
                payment: totalPayment,
                principal: principalPayment,
                interest: interestPayment,
                tranche: this.getTrancheForMonth(i, tranches)
            });
            
            // Break if debt is fully paid
            if (currentDebt <= 0) {
                break;
            }
        }
        
        // Calculate totals
        const totalPaymentAmount = paymentSchedule.reduce((sum, payment) => sum + payment.payment, 0);
        const totalInterest = paymentSchedule.reduce((sum, payment) => sum + payment.interest, 0);
        const overpayment = totalPaymentAmount - totalDebt;
        const effectiveRate = (overpayment / totalDebt) * 100;
        
        return {
            totalCreditAmount: totalDebt,
            totalPaymentAmount: totalPaymentAmount,
            overpaymentAmount: overpayment,
            effectiveRate: effectiveRate,
            paymentSchedule: paymentSchedule,
            tranches: tranches
        };
    }

    getTrancheForMonth(month, tranches) {
        // For this example, we'll return the first tranche
        // In a real implementation, this would determine which tranche is active for the given month
        return `Транш #1`;
    }

    displayResults(results) {
        // Update summary information
        document.getElementById('totalCreditAmount').textContent = this.formatCurrency(results.totalCreditAmount);
        document.getElementById('totalPaymentAmount').textContent = this.formatCurrency(results.totalPaymentAmount);
        document.getElementById('overpaymentAmount').textContent = this.formatCurrency(results.overpaymentAmount);
        document.getElementById('effectiveRate').textContent = results.effectiveRate.toFixed(2) + '%';
        
        // Update payment table
        this.updatePaymentTable(results.paymentSchedule);
        
        // Update charts
        this.updatePaymentChart(results.paymentSchedule);
        this.updateStructureChart(results.paymentSchedule);
        this.updateBalanceChart(results.paymentSchedule);
    }

    updatePaymentTable(schedule) {
        const tbody = document.querySelector('#paymentTable tbody');
        tbody.innerHTML = '';
        
        schedule.forEach(payment => {
            const row = document.createElement('tr');
            row.innerHTML = `
                <td>${payment.month}</td>
                <td>${payment.date}</td>
                <td>${this.formatCurrency(payment.balance)}</td>
                <td>${this.formatCurrency(payment.payment)}</td>
                <td>${this.formatCurrency(payment.principal)}</td>
                <td>${this.formatCurrency(payment.interest)}</td>
                <td>${payment.tranche}</td>
            `;
            tbody.appendChild(row);
        });
    }

    updatePaymentChart(schedule) {
        const ctx = document.getElementById('paymentChart').getContext('2d');
        
        if (this.paymentChart) {
            this.paymentChart.destroy();
        }
        
        const months = schedule.map((_, i) => `Месяц ${i + 1}`);
        const payments = schedule.map(p => p.payment);
        const interests = schedule.map(p => p.interest);
        const principals = schedule.map(p => p.principal);
        
        this.paymentChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: months,
                datasets: [
                    {
                        label: 'Общий платеж',
                        data: payments,
                        borderColor: '#2563eb',
                        backgroundColor: 'rgba(37, 99, 235, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'Проценты',
                        data: interests,
                        borderColor: '#ef4444',
                        backgroundColor: 'rgba(239, 68, 68, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3
                    },
                    {
                        label: 'Основной долг',
                        data: principals,
                        borderColor: '#10b981',
                        backgroundColor: 'rgba(16, 185, 129, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Ежемесячные платежи'
                    },
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return value.toLocaleString() + ' ₽';
                            }
                        }
                    },
                    x: {
                        display: false // Hide x-axis labels to save space
                    }
                }
            }
        });
    }

    updateStructureChart(schedule) {
        const ctx = document.getElementById('structureChart').getContext('2d');
        
        if (this.structureChart) {
            this.structureChart.destroy();
        }
        
        // Calculate totals
        const totalPrincipal = schedule.reduce((sum, p) => sum + p.principal, 0);
        const totalInterest = schedule.reduce((sum, p) => sum + p.interest, 0);
        
        this.structureChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: ['Основной долг', 'Проценты'],
                datasets: [{
                    data: [totalPrincipal, totalInterest],
                    backgroundColor: [
                        '#10b981',
                        '#ef4444'
                    ],
                    borderWidth: 0
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Структура выплат'
                    },
                    legend: {
                        position: 'bottom',
                    }
                }
            }
        });
    }

    updateBalanceChart(schedule) {
        const ctx = document.getElementById('balanceChart').getContext('2d');
        
        if (this.balanceChart) {
            this.balanceChart.destroy();
        }
        
        const months = schedule.map((_, i) => `Месяц ${i + 1}`);
        const balances = schedule.map(p => p.balance);
        
        this.balanceChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: months,
                datasets: [{
                    label: 'Остаток долга',
                    data: balances,
                    borderColor: '#8b5cf6',
                    backgroundColor: 'rgba(139, 92, 246, 0.1)',
                    borderWidth: 3,
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: true,
                        text: 'Остаток долга по месяцам'
                    },
                    legend: {
                        position: 'top',
                    }
                },
                scales: {
                    y: {
                        beginAtZero: false,
                        ticks: {
                            callback: function(value) {
                                return value.toLocaleString() + ' ₽';
                            }
                        }
                    },
                    x: {
                        display: false // Hide x-axis labels to save space
                    }
                }
            }
        });
    }

    formatCurrency(amount) {
        return new Intl.NumberFormat('ru-RU', {
            style: 'currency',
            currency: 'RUB',
            minimumFractionDigits: 0,
            maximumFractionDigits: 0
        }).format(amount);
    }

    resetCalculator() {
        // Reset form
        document.getElementById('mortgageForm').reset();
        
        // Remove all tranches except the first one
        const tranchesList = document.getElementById('tranchesList');
        while (tranchesList.firstChild) {
            tranchesList.removeChild(tranchesList.firstChild);
        }
        
        // Add initial tranche
        this.trancheCounter = 0;
        this.addTranche();
        
        // Hide dashboard
        document.getElementById('dashboard').style.display = 'none';
        
        // Reset charts
        if (this.paymentChart) {
            this.paymentChart.destroy();
            this.paymentChart = null;
        }
        
        if (this.structureChart) {
            this.structureChart.destroy();
            this.structureChart = null;
        }
        
        if (this.balanceChart) {
            this.balanceChart.destroy();
            this.balanceChart = null;
        }
        
        // Scroll to top
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

// Initialize calculator when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.calculator = new MortgageCalculator();
});